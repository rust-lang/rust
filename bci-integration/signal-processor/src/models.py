"""Pydantic response models matching the JSON schemas.

These models enforce type safety at the server boundary and auto-generate
OpenAPI documentation for the FastAPI endpoints.
"""

from pydantic import BaseModel, Field


class BrainState(BaseModel):
    primary: str
    confidence: float = Field(ge=0, le=1)
    secondary: list[dict] = Field(default_factory=list)


class Scores(BaseModel):
    attention: float = Field(ge=0, le=1, default=0.0)
    relaxation: float = Field(ge=0, le=1, default=0.0)
    cognitive_load: float = Field(ge=0, le=1, default=0.0)


class BandPowers(BaseModel):
    delta: float = Field(ge=0, default=0.0)
    theta: float = Field(ge=0, default=0.0)
    alpha: float = Field(ge=0, default=0.0)
    beta: float = Field(ge=0, default=0.0)
    gamma: float = Field(ge=0, default=0.0)


class PauseEventModel(BaseModel):
    """Represents a detected pause trigger."""
    pause_type: str  # "deliberate" | "automatic"
    trigger: str  # "jaw_clench" | "drowsiness" | "headset_removed"
    confidence: float = Field(ge=0, le=1)
    timestamp_unix_ms: int
    recommended_action: str  # "pause" | "slow_down" | "stop"


class ResumeEventModel(BaseModel):
    """Represents a resume from paused state."""
    timestamp_unix_ms: int
    reason: str


class BCIStateModel(BaseModel):
    """Matches bci_state.schema.json."""
    timestamp_unix_ms: int
    session_id: str = Field(max_length=128)
    device_id: str = Field(max_length=128)
    state: BrainState
    scores: Scores
    band_powers: BandPowers | None = None
    signal_quality: float = Field(ge=0, le=1)
    artifact_probability: float = Field(ge=0, le=1, default=0.0)
    staleness_ms: int = Field(ge=0, default=0)
    natural_language_summary: str = Field(max_length=512)
    pause_event: PauseEventModel | None = None
    resume_event: ResumeEventModel | None = None


class WebhookFilter(BaseModel):
    """Filter criteria for webhook notifications."""
    state_is: str | None = None
    score_below: dict[str, float] | None = None
    score_above: dict[str, float] | None = None


class WebhookRegistration(BaseModel):
    """Request body for registering a webhook."""
    url: str
    filters: WebhookFilter | None = None


class WebhookInfo(BaseModel):
    """Info about a registered webhook."""
    id: str
    url: str
    filters: WebhookFilter | None = None
    created_at: int


class WebhookEvent(BaseModel):
    """Payload sent to webhook endpoints."""
    event: str
    previous_state: str
    new_state: str
    bci_state: dict
    timestamp_unix_ms: int


class StateResponse(BaseModel):
    """Matches state_server_api.schema.json#/definitions/state_response."""
    available: bool
    timestamp_unix_ms: int
    bci_state: BCIStateModel | None = None
    error: str | None = Field(default=None, max_length=512)


class HealthResponse(BaseModel):
    """Matches state_server_api.schema.json#/definitions/health_response."""
    status: str  # "ok" | "degraded" | "no_signal"
    uptime_seconds: float
    device_connected: bool = False
    session_id: str | None = None
    last_update_unix_ms: int | None = None
