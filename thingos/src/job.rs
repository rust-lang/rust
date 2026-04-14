//! Canonical public types for the `thingos.job` schema kind.
//!
//! # Schema
//!
//! ## v1 (Phase 2) — lifecycle shape
//!
//! ```text
//! kind thingos.job.state = enum {
//!   New,
//!   Running,
//!   Exited,
//! }
//!
//! kind thingos.job = struct {
//!   state: thingos.job.state,
//! }
//! ```
//!
//! ## v2 (Phase 3) — exit and wait shapes
//!
//! ```text
//! kind thingos.job.exit = struct {
//!   state: thingos.job.state,
//!   code:  option<i32>,
//! }
//!
//! kind thingos.job.wait.result = enum {
//!   Running,
//!   Exited { code: option<i32> },
//! }
//! ```
//!
//! # Transitional mapping
//!
//! The current kernel `Process` structure is the *provisional* internal
//! backing for a canonical `Job`.  A dedicated bridge layer
//! (`kernel::job::bridge`) converts `Process` lifecycle state into this public
//! representation so that the new ontology appears first at the edges while
//! internal machinery is replaced gradually.
//!
//! The `Process` struct is deliberately *not* renamed or split here.  Future
//! phases will hollow it out by migrating individual responsibilities:
//!
//! * **Phase 3** (this phase): wait/exit reporting migrates into `Job` terms.
//! * **Phase 4**: introduce `Group` as the coordination truth.
//! * **Phase 5**: begin internal `Process` decomposition by responsibility.
//!
//! | Canonical field         | Current kernel source                                      |
//! |-------------------------|------------------------------------------------------------|
//! | `Job::state`            | inferred from the live `ThreadState` of the thread group   |
//! | `JobExit::state`        | `ThreadState::Dead` → `JobState::Exited`                   |
//! | `JobExit::code`         | `Thread::exit_code` (set by `mark_task_exited`)            |
//! | `JobWaitResult::Exited` | result of `poll_task_exit` / `waitpid` returning Some(code)|
//! | `JobWaitResult::Running`| result of `poll_task_exit` returning `None`                |
//!
//! # Note on `Process` vs `Job`
//!
//! Not all of `Process` will eventually become `Job`.  `Process` currently
//! carries address-space ownership, FD tables, namespaces, signal state, and
//! more.  Those responsibilities will migrate to their own canonical kinds
//! over time.  `Job` covers *only* the lifecycle/accounting axis: creation,
//! running, and exit.

/// Canonical lifecycle state for a `thingos.job`.
///
/// This is the external truth for process lifecycle exposed at system
/// boundaries.  The kernel's internal `Process`+`Thread` model maps into this
/// via `kernel::job::bridge`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JobState {
    /// The job has been created but no threads are yet running.
    New,
    /// At least one thread in the job is alive (Runnable, Running, or Blocked).
    Running,
    /// All threads in the job have exited.
    Exited,
}

impl JobState {
    /// Return a short human-readable label.
    pub fn as_str(self) -> &'static str {
        match self {
            JobState::New => "New",
            JobState::Running => "Running",
            JobState::Exited => "Exited",
        }
    }
}

/// Canonical public representation of a lifecycle/accounting container.
///
/// Corresponds to the `thingos.job` schema kind (v1).  The kernel's internal
/// `Process` structure is the current transitional backing; the `bridge` module
/// in `kernel::job` converts `Process`-shaped state into this type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Job {
    /// The current lifecycle state of this job.
    pub state: JobState,
}

// ── Phase 3: exit and wait types ─────────────────────────────────────────────

/// Canonical description of an exited (or still-live) job.
///
/// Corresponds to the `thingos.job.exit` schema kind (v1).
///
/// When `state` is `JobState::Exited`, `code` holds the integer exit code
/// reported by the process.  For live jobs (`New` or `Running`) `code` is
/// `None`.
///
/// # Transitional mapping
///
/// The current kernel sets `Thread::exit_code` inside `mark_task_exited`.
/// `kernel::job::bridge::job_exit_from_snapshot` reads that field (via
/// `ProcessSnapshot`) to populate this type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JobExit {
    /// Lifecycle state at the time of the snapshot.
    pub state: JobState,
    /// Exit code, present only when `state == JobState::Exited`.
    pub code: Option<i32>,
}

impl JobExit {
    /// Format as a human-readable text blob suitable for procfs.
    ///
    /// Outputs two lines:
    /// ```text
    /// state: Running
    /// code: -
    /// ```
    /// or
    /// ```text
    /// state: Exited
    /// code: 0
    /// ```
    pub fn as_text(&self) -> alloc::string::String {
        alloc::format!(
            "state: {}\ncode: {}\n",
            self.state.as_str(),
            match self.code {
                Some(c) => alloc::format!("{}", c),
                None => alloc::string::String::from("-"),
            }
        )
    }
}

/// The canonical result of waiting on (or polling) a job.
///
/// Corresponds to the `thingos.job.wait.result` schema kind (v1).
///
/// # Transitional mapping
///
/// * `poll_task_exit` returning `None`       → `JobWaitResult::Running`
/// * `poll_task_exit` returning `Some(code)` → `JobWaitResult::Exited { code: Some(code) }`
/// * `waitpid` returning `(pid, code)`       → `JobWaitResult::Exited { code: Some(code) }`
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JobWaitResult {
    /// The job is still running; no exit code available yet.
    Running,
    /// The job has exited.  `code` is the integer exit code (always `Some`
    /// when produced by the current kernel, which always records a code).
    Exited {
        /// The exit code reported by the process.
        code: Option<i32>,
    },
}

impl JobWaitResult {
    /// Return a short label for the state part of the result.
    pub fn state_str(&self) -> &'static str {
        match self {
            JobWaitResult::Running => "Running",
            JobWaitResult::Exited { .. } => "Exited",
        }
    }

    /// Format as a human-readable text blob suitable for procfs.
    pub fn as_text(&self) -> alloc::string::String {
        match self {
            JobWaitResult::Running => alloc::string::String::from("result: Running\ncode: -\n"),
            JobWaitResult::Exited { code } => alloc::format!(
                "result: Exited\ncode: {}\n",
                match code {
                    Some(v) => alloc::format!("{}", v),
                    None => alloc::string::String::from("-"),
                }
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── JobState ─────────────────────────────────────────────────────────────

    #[test]
    fn test_job_state_as_str() {
        assert_eq!(JobState::New.as_str(), "New");
        assert_eq!(JobState::Running.as_str(), "Running");
        assert_eq!(JobState::Exited.as_str(), "Exited");
    }

    #[test]
    fn test_job_state_equality() {
        assert_eq!(JobState::New, JobState::New);
        assert_ne!(JobState::New, JobState::Running);
        assert_ne!(JobState::Running, JobState::Exited);
    }

    // ── Job ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_job_carries_state() {
        let job = Job { state: JobState::Running };
        assert_eq!(job.state, JobState::Running);
    }

    // ── JobExit ──────────────────────────────────────────────────────────────

    #[test]
    fn test_job_exit_as_text_running() {
        let exit = JobExit { state: JobState::Running, code: None };
        let text = exit.as_text();
        assert!(text.contains("state: Running"), "unexpected: {}", text);
        assert!(text.contains("code: -"), "unexpected: {}", text);
    }

    #[test]
    fn test_job_exit_as_text_exited_with_code() {
        let exit = JobExit { state: JobState::Exited, code: Some(0) };
        let text = exit.as_text();
        assert!(text.contains("state: Exited"), "unexpected: {}", text);
        assert!(text.contains("code: 0"), "unexpected: {}", text);
    }

    #[test]
    fn test_job_exit_as_text_exited_nonzero() {
        let exit = JobExit { state: JobState::Exited, code: Some(42) };
        let text = exit.as_text();
        assert!(text.contains("code: 42"), "unexpected: {}", text);
    }

    // ── JobWaitResult ────────────────────────────────────────────────────────

    #[test]
    fn test_job_wait_result_state_str() {
        assert_eq!(JobWaitResult::Running.state_str(), "Running");
        assert_eq!(JobWaitResult::Exited { code: Some(0) }.state_str(), "Exited");
        assert_eq!(JobWaitResult::Exited { code: None }.state_str(), "Exited");
    }

    #[test]
    fn test_job_wait_result_as_text_running() {
        let text = JobWaitResult::Running.as_text();
        assert!(text.contains("result: Running"), "unexpected: {}", text);
        assert!(text.contains("code: -"), "unexpected: {}", text);
    }

    #[test]
    fn test_job_wait_result_as_text_exited_with_code() {
        let text = JobWaitResult::Exited { code: Some(1) }.as_text();
        assert!(text.contains("result: Exited"), "unexpected: {}", text);
        assert!(text.contains("code: 1"), "unexpected: {}", text);
    }

    #[test]
    fn test_job_wait_result_as_text_exited_no_code() {
        let text = JobWaitResult::Exited { code: None }.as_text();
        assert!(text.contains("result: Exited"), "unexpected: {}", text);
        assert!(text.contains("code: -"), "unexpected: {}", text);
    }

    #[test]
    fn test_job_wait_result_equality() {
        assert_eq!(JobWaitResult::Running, JobWaitResult::Running);
        assert_ne!(JobWaitResult::Running, JobWaitResult::Exited { code: Some(0) });
        assert_eq!(
            JobWaitResult::Exited { code: Some(0) },
            JobWaitResult::Exited { code: Some(0) }
        );
        assert_ne!(
            JobWaitResult::Exited { code: Some(0) },
            JobWaitResult::Exited { code: Some(1) }
        );
    }
}
