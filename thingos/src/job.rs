//! Canonical public types for the `thingos.job` schema kind.
//!
//! # Schema (v1)
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
//! * **Phase 3**: wait/exit reporting migrates into `Job` terms.
//! * **Phase 4**: introduce `Group` as the coordination truth.
//! * **Phase 5**: begin internal `Process` decomposition by responsibility.
//!
//! | Canonical field | Current kernel source                              |
//! |-----------------|----------------------------------------------------|
//! | `state`         | inferred from the live `ThreadState` of the group  |
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
///
/// This struct is intentionally minimal for v1.  Exit codes, resource
/// accounting, and wait semantics will be added in Phase 3 once the wait/exit
/// path migrates into Job terms.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Job {
    /// The current lifecycle state of this job.
    pub state: JobState,
}
