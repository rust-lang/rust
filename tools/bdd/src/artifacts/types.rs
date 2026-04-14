use std::path::PathBuf;

/// Result of a step execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepResult {
    Passed,
    Failed,
    Skipped,
}

impl StepResult {
    pub fn emoji(&self) -> &'static str {
        match self {
            StepResult::Passed => "✅",
            StepResult::Failed => "❌",
            StepResult::Skipped => "⏭️",
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            StepResult::Passed => "passed",
            StepResult::Failed => "failed",
            StepResult::Skipped => "skipped",
        }
    }
}

/// Captures details about a single step execution.
#[derive(Debug, Clone)]
pub struct StepArtifacts {
    pub name: String,
    pub keyword: String,
    pub result: StepResult,
    pub dir: PathBuf,
    pub screenshot_before: Option<PathBuf>,
    pub screenshot_after: Option<PathBuf>,
    pub registers: Option<PathBuf>,
    pub serial_log: Option<PathBuf>,
    pub serial_excerpt: String,
    pub duration_ms: u64,
}

/// Captures details about a scenario execution.
#[derive(Debug, Clone)]
pub struct ScenarioArtifacts {
    pub name: String,
    pub dir: PathBuf,
    pub steps: Vec<StepArtifacts>,
    pub passed: bool,
}

/// Captures details about a feature execution.
#[derive(Debug, Clone)]
pub struct FeatureArtifacts {
    pub name: String,
    pub dir: PathBuf,
    pub scenarios: Vec<ScenarioArtifacts>,
}
