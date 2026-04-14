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
        let code_str = match self.code {
            Some(c) => alloc::format!("{}", c),
            None => alloc::string::String::from("-"),
        };
        alloc::format!("state: {}\ncode: {}\n", self.state.as_str(), code_str)
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
            JobWaitResult::Exited { code } => {
                let c = match code {
                    Some(v) => alloc::format!("{}", v),
                    None => alloc::string::String::from("-"),
                };
                alloc::format!("result: Exited\ncode: {}\n", c)
            }
        }
    }
}
