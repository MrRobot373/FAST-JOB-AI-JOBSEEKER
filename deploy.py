import os
import json
import asyncio
import logging
import time
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
import hashlib
import uuid

from fastapi import FastAPI, HTTPException, Request, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import uvicorn
import google.generativeai as genai

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Global Metrics Tracking
# -----------------------------------------------------------------------------
class MetricsTracker:
    def __init__(self):
        self.active_requests = 0
        self.total_requests = 0
        self.error_count = 0
        self.start_time = time.time()
    
    def request_started(self):
        self.active_requests += 1
        self.total_requests += 1
    
    def request_completed(self):
        self.active_requests = max(0, self.active_requests - 1)
    
    def error_occurred(self):
        self.error_count += 1
    
    def get_stats(self):
        uptime = time.time() - self.start_time
        return {
            "active_requests": self.active_requests,
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "uptime_seconds": round(uptime, 2),
            "requests_per_minute": round((self.total_requests / uptime) * 60, 2) if uptime > 0 else 0
        }

metrics = MetricsTracker()

# -----------------------------------------------------------------------------
# Application Lifespan Events
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Job Seeker API starting up...")
    logger.info(f"📊 Workers: {os.getenv('WORKERS', '4')}")
    logger.info(f"🔑 Gemini API configured: {'✅' if os.getenv('GEMINI_API_KEY') else '❌'}")
    logger.info(f"⚡ Max concurrent Gemini requests: {os.getenv('MAX_CONCURRENT_REQUESTS', '50')}")
    yield
    # Shutdown
    logger.info("🛑 Job Seeker API shutting down...")
    logger.info(f"📈 Final stats: {metrics.get_stats()}")

# -----------------------------------------------------------------------------
# Enhanced Gemini API Manager with True Async Support
# -----------------------------------------------------------------------------
class GeminiAPIManager:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("⚠️ GEMINI_API_KEY not set, using fallback (not recommended for production)")
            api_key = "AIzaSyB_PmrNCqr11BOXukZEqfX8Q0f16UYtGwE"
        
        genai.configure(api_key=api_key)
        
        self.generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-lite",
            generation_config=self.generation_config,
        )
        
        # Enhanced concurrency control
        self.max_concurrent_requests = int(os.getenv('MAX_CONCURRENT_REQUESTS', '50'))
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self.request_count = 0
        self.active_requests = 0
        
        logger.info(f"🎯 Gemini API Manager initialized with {self.max_concurrent_requests} concurrent slots")
    
    async def generate_content_async(self, prompt: str, timeout: int = 45, request_id: str = None) -> str:
        """Enhanced async wrapper for Gemini API with detailed logging"""
        request_id = request_id or str(uuid.uuid4())[:8]
        
        async with self.semaphore:
            self.active_requests += 1
            self.request_count += 1
            
            logger.info(f"🤖 [{request_id}] Starting Gemini API call ({self.active_requests}/{self.max_concurrent_requests} active)")
            
            try:
                start_time = time.time()
                
                # Use asyncio.to_thread for true async behavior
                response = await asyncio.wait_for(
                    asyncio.to_thread(self._sync_generate, prompt, request_id),
                    timeout=timeout
                )
                
                processing_time = time.time() - start_time
                logger.info(f"✅ [{request_id}] Gemini API completed in {processing_time:.2f}s")
                
                return response
                
            except asyncio.TimeoutError:
                logger.error(f"⏰ [{request_id}] Gemini API timeout after {timeout}s")
                raise HTTPException(
                    status_code=status.HTTP_408_REQUEST_TIMEOUT,
                    detail=f"AI processing timeout after {timeout} seconds"
                )
            except Exception as e:
                logger.error(f"❌ [{request_id}] Gemini API error: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"AI service temporarily unavailable: {str(e)}"
                )
            finally:
                self.active_requests -= 1
    
    def _sync_generate(self, prompt: str, request_id: str) -> str:
        """Synchronous Gemini API call with enhanced error handling"""
        try:
            chat_session = self.model.start_chat(history=[])
            response = chat_session.send_message(prompt)
            return response.text
        except Exception as e:
            logger.error(f"🔥 [{request_id}] Sync generation error: {str(e)}")
            raise e
    
    def get_status(self):
        """Get current API manager status"""
        return {
            "max_concurrent": self.max_concurrent_requests,
            "active_requests": self.active_requests,
            "available_slots": self.max_concurrent_requests - self.active_requests,
            "total_requests": self.request_count,
            "utilization_percent": round((self.active_requests / self.max_concurrent_requests) * 100, 1)
        }

# Initialize Gemini manager
gemini_manager = GeminiAPIManager()

# -----------------------------------------------------------------------------
# Enhanced Pydantic Models with Comprehensive Validation
# -----------------------------------------------------------------------------
class ResumeData(BaseModel):
    name: str = "User"
    skills: list = []
    experience: str = "Not provided"
    education: str = "Not provided"
    otherDetails: dict = {}
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) < 1:
            raise ValueError('Name cannot be empty')
        if len(v) > 100:
            raise ValueError('Name too long (max 100 characters)')
        return v.strip()
    
    @validator('skills')
    def validate_skills(cls, v):
        if len(v) > 100:
            raise ValueError('Too many skills (max 100)')
        return v
    
    @validator('otherDetails')
    def validate_other_details(cls, v):
        json_str = json.dumps(v)
        if len(json_str) > 100000:  # 100KB limit
            raise ValueError('Resume data too large (max 100KB)')
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class QAGeneratorRequest(BaseModel):
    resume: dict
    jd: Optional[Dict] = None
    
    @validator('resume', 'jd')
    def validate_data_size(cls, v):
        if v and len(json.dumps(v)) > 75000:
            raise ValueError('Data too large (max 75KB)')
        return v

class ResumeUpdaterRequest(BaseModel):
    resume: dict
    jd: dict
    
    @validator('resume', 'jd')
    def validate_data_size(cls, v):
        if len(json.dumps(v)) > 75000:
            raise ValueError('Data too large (max 75KB)')
        return v

# -----------------------------------------------------------------------------
# User Identification for Parallel Processing
# -----------------------------------------------------------------------------
def generate_user_id(request_data: str, ip_address: str = None, user_agent: str = None) -> str:
    """Generate consistent user ID for tracking parallel requests"""
    identifier_string = f"{request_data}{ip_address or 'unknown'}{user_agent or 'unknown'}"
    return hashlib.md5(identifier_string.encode()).hexdigest()[:12]

# -----------------------------------------------------------------------------
# Async Processing Functions
# -----------------------------------------------------------------------------
async def analyze_resume_with_gemini_async(resume_data: dict, request_id: str = None) -> str:
    """Async resume analysis with parallel processing support"""
    resume_json_str = json.dumps(resume_data, indent=2)

    prompt = f"""
    You are a highly skilled AI Resume Analyst. Carefully analyze the provided resume
    (in JSON format below) and return **a structured, well-formatted response** like a professional resume consultant.

    **🔍 Your Task:**
    - Title the Resume** (e.g., "Confident", "Needs Improvement", "Underconfident").
    - Rewrite the Summary Professionally** (Highlight skills and impact). Ensure it's concise (around 70 words).
    - Suggest Relevant Job Roles** (5-7 roles with a brief reason).
    - List Additional Skills to Learn** (Practical, relevant, and industry-specific).
    - Rate the Resume Level** (Beginner, Intermediate, Advanced + one-line explanation).
    - Point Out Resume Improvements** (Structured sections with actionable feedback).
    - Check for Grammar Mistakes** (Only mention if mistakes exist).

    **📌 Candidate Resume Data (JSON)**:
    {resume_json_str}

    ** Format Your Response EXACTLY Like This** (keeping these headings):

   Resume Title: <Your Title>

     Professional Summary:
    <Rewrite ONLY the summary from the resume's "otherDetails.summary" in a professional manner, and keep it short (around 70 words).>

     Suggested Job Role:
    1. <Role> - [Reason]
    2. <Role> - [Reason]
    ...
   
     Additional Skills to Learn:
    - <Skill> - [Why it is important]
    - <Skill> - [Why it is important]
    ...

     Resume Rating: <Beginner/Intermediate/Advanced>
    - <One-line explanation.>

     Resume Improvement:
    - Skills Section: "<Actionable improvement>"
    - Experience Section: "<Actionable improvement>"
    - Certifications Section: "<Actionable improvement>"
    ...

     Grammar Check:
    - <Only mention if mistakes exist; otherwise, skip.>

    **⚠️ Important:**  
    - DO NOT generate made-up job roles or skills.  
    - Base your analysis ONLY on the provided JSON data.  
    - Keep responses clear, formatted, and actionable.
    - DO Not use brackets in responses  
    """
    
    return await gemini_manager.generate_content_async(prompt, timeout=45, request_id=request_id)

async def generate_questions_with_answers_async(resume: dict, jd: Optional[dict] = None, request_id: str = None) -> str:
    """Async Q&A generation with parallel processing"""
    jd_section = f"\nJob Description:\n{json.dumps(jd, indent=2)}" if jd else ""
    source_instruction = (
        "Blend Resume and JD to generate realistic and relevant questions."
        if jd else
        "Use only the Resume to generate questions."
    )

    prompt = f"""
You are a senior interviewer preparing a mock interview for a candidate.

🎯 Based on the resume{' and job description' if jd else ''}, infer the most suitable role or domain.
Then generate **20 interview questions with full, descriptive answers**:
- For technical roles: prioritize domain-specific 70% technical questions 30% Non-technical questions.
- For non-technical roles: include a mix of domain-relevant and behavioral questions.

✅ Format:
- Use clear section headers as needed (Technical, Non Technical) for technical roles only
- Number each Q&A from **1 to 20**
- For each:
  [number]. Q: [question]
     A: [answer]

- Ensure Q and A are on separate lines
- Infer answers professionally even if information is limited
{source_instruction}

{jd_section}

Resume:
{json.dumps(resume, indent=2)}

Now generate the full response.
"""
    
    return await gemini_manager.generate_content_async(prompt, timeout=50, request_id=request_id)

async def expand_jd_skills_async(jd_skills: set, request_id: str = None) -> dict:
    """Async JD skills expansion"""
    prompt = f"""
Given the following job description (JD) skills, generate a JSON dictionary where each JD skill
is a key, and the value is a list of relevant or commonly associated skills.

JD Skills: {json.dumps(list(jd_skills))}

Return output as a valid JSON dictionary, nothing else.

Example Output:
{{
    "React": ["Redux", "Angular", "Next.js", "Node.js"],
    "Python": ["Django", "Flask", "FastAPI", "TensorFlow"]
}}
"""
    try:
        response = await gemini_manager.generate_content_async(prompt, timeout=30, request_id=request_id)
        return json.loads(response.strip())
    except Exception as e:
        logger.warning(f"🔄 [{request_id}] Skill expansion failed: {e}, using fallback")
        return {skill: [skill] for skill in jd_skills}

async def reorder_resume_by_skill_match_async(jd: dict, resume: dict, request_id: str = None) -> dict:
    """Async resume reordering with skill matching"""
    jd_skills = {s.lower().strip() for s in jd.get("skills", [])}
    
    if jd_skills:
        expanded_skill_map = await expand_jd_skills_async(jd_skills, request_id)
        expanded_jd_skills = set(jd_skills)
        for related_skills in expanded_skill_map.values():
            expanded_jd_skills.update(s.lower().strip() for s in related_skills)
    else:
        expanded_jd_skills = set()

    def count_skill_matches(skill_list):
        if not skill_list or not expanded_jd_skills:
            return 0
        return sum(skill.lower() in expanded_jd_skills for skill in skill_list)

    # Process resume sections in parallel where possible
    resume_copy = resume.copy()

    # Reorder certifications
    if "certificationDetails" in resume_copy and isinstance(resume_copy["certificationDetails"], list):
        resume_copy["certificationDetails"] = sorted(
            resume_copy["certificationDetails"],
            key=lambda cert: count_skill_matches(cert.get("skills", [])),
            reverse=True
        )

    # Reorder professional details
    if "professionalDetails" in resume_copy and isinstance(resume_copy["professionalDetails"], list):
        resume_copy["professionalDetails"] = sorted(
            resume_copy["professionalDetails"],
            key=lambda prof: count_skill_matches(prof.get("skills", [])),
            reverse=True
        )

    # Reorder project details
    if "projectDetails" in resume_copy and isinstance(resume_copy["projectDetails"], list):
        resume_copy["projectDetails"] = sorted(
            resume_copy["projectDetails"],
            key=lambda proj: count_skill_matches(proj.get("skills", [])),
            reverse=True
        )

    # Reorder skills
    if ("otherDetails" in resume_copy and isinstance(resume_copy["otherDetails"], dict) and
        "skills" in resume_copy["otherDetails"] and isinstance(resume_copy["otherDetails"]["skills"], list)):
        resume_copy["otherDetails"]["skills"] = sorted(
            resume_copy["otherDetails"]["skills"],
            key=lambda skill: skill.lower() in expanded_jd_skills,
            reverse=True
        )

    return resume_copy

async def rewrite_summary_async(original_summary: str, jd: dict, request_id: str = None) -> str:
    """Async summary rewriting with enhanced error handling"""
    prompt = f"""
Rewrite and enhance the following resume summary to better align with the job description (JD).
- Keep all important details from the original summary.
- Expand and integrate key job requirements from the JD.
- Maintain professional, structured, and fluent writing.
- Naturally include JD-specific skills as per JD and resume describing the responsibilities, and industry context.
- Keep it short (around 70 words), and ensure the summary does not mention skills as an array (i.e., no brackets).
- Do NOT shorten, add, or remove content, only enhance and improve relevance.

### Job Description:
{json.dumps(jd, indent=2)}

### Original Resume Summary:
{original_summary}

### Enhanced Resume Summary (with JD Integration):
"""
    try:
        response = await gemini_manager.generate_content_async(prompt, timeout=35, request_id=request_id)
        new_summary = response.strip()

        # Ensure the summary is around 70 words
        words = new_summary.split()
        if len(words) > 70:
            return ' '.join(words[:70])

        return new_summary
    except Exception as e:
        logger.warning(f"📝 [{request_id}] Summary rewrite failed: {e}, using original")
        return original_summary

async def enhance_summary_async(jd: dict, resume: dict, request_id: str = None) -> dict:
    """Async summary enhancement"""
    resume_copy = resume.copy()
    
    if ("otherDetails" in resume_copy and isinstance(resume_copy["otherDetails"], dict) and
        "summary" in resume_copy["otherDetails"] and isinstance(resume_copy["otherDetails"]["summary"], str)):
        
        original_summary = resume_copy["otherDetails"]["summary"]
        new_summary = await rewrite_summary_async(original_summary, jd, request_id)
        resume_copy["otherDetails"]["summary"] = new_summary
    
    return resume_copy

# -----------------------------------------------------------------------------
# FastAPI Setup with Enhanced Configuration
# -----------------------------------------------------------------------------
app = FastAPI(
    title="JOB_SEEKER_API_V2",
    description="High-performance parallel processing job seeker tools with AI analysis",
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-ID"]
)

# -----------------------------------------------------------------------------
# Enhanced Middleware for Request Tracking and Performance
# -----------------------------------------------------------------------------
@app.middleware("http")
async def enhanced_request_middleware(request: Request, call_next):
    # Generate unique request ID
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    
    start_time = time.time()
    metrics.request_started()
    
    # Enhanced request logging
    user_agent = request.headers.get("user-agent", "unknown")
    logger.info(f"📥 [{request_id}] {request.method} {request.url.path} - "
               f"Client: {request.client.host} - Active: {metrics.active_requests}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Success logging
        logger.info(f"✅ [{request_id}] {request.method} {request.url.path} - "
                   f"{response.status_code} - {process_time:.3f}s")
        
        # Add performance headers
        response.headers["X-Process-Time"] = f"{process_time:.3f}"
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Active-Requests"] = str(metrics.active_requests)
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        metrics.error_occurred()
        
        logger.error(f"❌ [{request_id}] {request.method} {request.url.path} - "
                    f"Error: {str(e)[:100]} - {process_time:.3f}s")
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred during processing",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "path": str(request.url.path)
            },
            headers={
                "X-Process-Time": f"{process_time:.3f}",
                "X-Request-ID": request_id
            }
        )
    finally:
        metrics.request_completed()

# -----------------------------------------------------------------------------
# Enhanced Health Check and Monitoring Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "job-seeker-api",
        "version": "2.1.0",
        "parallel_processing": "enabled"
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Comprehensive health check with performance metrics"""
    gemini_status = gemini_manager.get_status()
    app_metrics = metrics.get_stats()
    
    # Test Gemini API connectivity
    api_test_status = "unknown"
    try:
        test_response = await asyncio.wait_for(
            gemini_manager.generate_content_async("Health check test", timeout=5),
            timeout=6
        )
        api_test_status = "healthy" if test_response else "unhealthy"
    except Exception as e:
        api_test_status = f"unhealthy: {str(e)[:50]}"
    
    overall_status = "healthy"
    if api_test_status.startswith("unhealthy") or app_metrics["error_count"] > 10:
        overall_status = "degraded"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "service": "job-seeker-api",
        "version": "2.1.0",
        "parallel_processing": {
            "enabled": True,
            "gemini_api": gemini_status,
            "api_connectivity": api_test_status
        },
        "performance_metrics": app_metrics,
        "system_info": {
            "workers": os.getenv("WORKERS", "4"),
            "max_concurrent_requests": gemini_manager.max_concurrent_requests,
            "memory_limit": "4Gi",
            "cpu_limit": "2"
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Detailed metrics endpoint for monitoring"""
    return {
        "application_metrics": metrics.get_stats(),
        "gemini_api_metrics": gemini_manager.get_status(),
        "timestamp": datetime.utcnow().isoformat()
    }

# -----------------------------------------------------------------------------
# Enhanced Async Endpoints with Parallel Processing
# -----------------------------------------------------------------------------
@app.post("/ANALYSER")
async def analyze_resume_api(
    resume_data: ResumeData,
    request: Request,
    x_forwarded_for: str = Header(None),
    user_agent: str = Header(None)
):
    """Enhanced async resume analysis with parallel processing and tracking"""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4())[:8])
    
    try:
        logger.info(f"🔍 [{request_id}] Starting resume analysis for: {resume_data.name}")
        
        # Generate user ID for tracking
        user_id = generate_user_id(
            str(resume_data.dict()), 
            x_forwarded_for or request.client.host,
            user_agent
        )
        
        resume_dict = resume_data.dict()
        analysis_result = await analyze_resume_with_gemini_async(resume_dict, request_id)
        
        logger.info(f"✅ [{request_id}] Resume analysis completed for: {resume_data.name}")
        
        return {
            "analysis": analysis_result,
            "request_id": request_id,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success",
            "processing_mode": "parallel"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [{request_id}] Resume analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/QUESTIONEER")
async def generate_qa_api(
    request_data: QAGeneratorRequest,
    request: Request,
    x_forwarded_for: str = Header(None),
    user_agent: str = Header(None)
):
    """Enhanced async Q&A generation with parallel processing"""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4())[:8])
    
    try:
        logger.info(f"❓ [{request_id}] Starting Q&A generation")
        
        # Generate user ID
        user_id = generate_user_id(
            str(request_data.dict()),
            x_forwarded_for or request.client.host,
            user_agent
        )
        
        result = await generate_questions_with_answers_async(
            request_data.resume, 
            request_data.jd, 
            request_id
        )
        
        logger.info(f"✅ [{request_id}] Q&A generation completed")
        
        return {
            "qa_output": result,
            "request_id": request_id,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success",
            "processing_mode": "parallel"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [{request_id}] Q&A generation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Q&A generation failed: {str(e)}"
        )

@app.post("/RESUME_GENERATOR")
async def resume_updater(
    request_data: ResumeUpdaterRequest,
    request: Request,
    x_forwarded_for: str = Header(None),
    user_agent: str = Header(None)
):
    """Enhanced async resume enhancement with parallel processing"""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4())[:8])
    
    try:
        logger.info(f"🔄 [{request_id}] Starting resume enhancement")
        
        # Generate user ID
        user_id = generate_user_id(
            str(request_data.dict()),
            x_forwarded_for or request.client.host,
            user_agent
        )
        
        # Process both operations concurrently for maximum parallelism
        reorder_task = reorder_resume_by_skill_match_async(
            request_data.jd, 
            request_data.resume, 
            request_id
        )
        
        # Wait for reordering to complete, then enhance summary
        reordered = await reorder_task
        enhanced = await enhance_summary_async(request_data.jd, reordered, request_id)
        
        logger.info(f"✅ [{request_id}] Resume enhancement completed")
        
        return {
            "updated_resume": enhanced,
            "request_id": request_id,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success",
            "processing_mode": "parallel"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [{request_id}] Resume enhancement failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resume enhancement failed: {str(e)}"
        )

# -----------------------------------------------------------------------------
# Global Exception Handlers
# -----------------------------------------------------------------------------
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.warning(f"⚠️ [{request_id}] Validation error on {request.url.path}: {str(exc)}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": str(exc),
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        },
        headers={"X-Request-ID": request_id}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.error(f"💥 [{request_id}] Unhandled exception on {request.url.path}: {str(exc)}")
    metrics.error_occurred()
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred during processing",
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        },
        headers={"X-Request-ID": request_id}
    )

# -----------------------------------------------------------------------------
# Production Uvicorn Configuration with Optimal Settings
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    config = {
        "app": "main:app",
        "host": "0.0.0.0",
        "port": int(os.getenv("PORT", 8000)),
        "workers": int(os.getenv("WORKERS", 4)),
        "worker_class": "uvicorn.workers.UvicornWorker",
        "loop": "uvloop",  # High-performance event loop
        "http": "httptools",  # High-performance HTTP parser
        "access_log": True,
        "log_level": "info",
        "timeout_keep_alive": 30,
        "timeout_graceful_shutdown": 120,
        "limit_max_requests": 1000,
        "limit_concurrency": 1000,
        "backlog": 2048
    }
    
    logger.info(f"🚀 Starting Job Seeker API with configuration: {config}")
    uvicorn.run(**config)
