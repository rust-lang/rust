//! Canonical public types for the `thingos.task` schema kind.
//!
//! # Schema (v2)
//!
//! ```text
//! kind thingos.task.state = enum {
//!   New,
//!   Ready,
//!   Running,
//!   Blocked,
//!   Exited,
//! }
//!
//! kind thingos.task = struct {
//!   state: thingos.task.state,
//!   job:   option<ref<thingos.job>>,
//!   name:  option<string>,
//! }
//! ```
//!
//! # Transitional mapping
//!
//! The current kernel `Thread` structure is the *provisional* internal
//! backing for a canonical `Task`.  A dedicated bridge layer
//! (`kernel::task::bridge`) converts `Thread` state into this public
//! representation so that the new ontology appears first at the edges
//! while the internal machinery is replaced gradually.
//!
//! | Canonical field | Current kernel source                     |
//! |-----------------|-------------------------------------------|
//! | `state`         | `Thread::state` (ThreadState)             |
//! | `job`           | `Process::pid` (provisional Job ID / PID) |
//! | `name`          | `Thread::name` (human-readable label)     |
//!
//! `ThreadState::Runnable` maps to `TaskState::Ready` because the kernel
//! distinguishes "eligible to run" from "currently executing", which aligns
//! with the canonical `Ready`/`Running` split.
//!
//! ## Field optionality
//!
//! `job` is `None` for kernel-only threads that have no associated `Process`
//! (i.e. no Job backing exists yet).  `name` is `None` when no human-readable
//! label has been set via `set_current_task_name`.
//!
//! ## Intentional ahead-of-implementation shape
//!
//! The canonical public `Task` shape is intentionally richer than what the
//! internal janix substrate currently provides.  The `job` field references a
//! `thingos.job` kind whose authoritative Rust backing is still the transitional
//! `Process` struct; `name` is sourced from the thread-level `[u8; 32]` store
//! that exists today.  Both fields are optional so that the absence of a full
//! internal implementation is represented explicitly rather than silently.

/// Canonical execution lifecycle state for a `thingos.task`.
///
/// This is the external truth exposed at system boundaries (procfs, introspection
/// syscalls, debug output).  The kernel's internal `ThreadState` maps into this
/// via `kernel::task::bridge`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TaskState {
    /// Allocated but not yet eligible to run.
    New,
    /// Eligible to run; waiting for CPU time.
    Ready,
    /// Currently executing on a CPU.
    Running,
    /// Waiting for an event (I/O, lock, sleep, signal).
    Blocked,
    /// Has exited; exit code is available.
    Exited,
}

impl TaskState {
    /// Return a short human-readable label compatible with `/proc` status files.
    pub fn as_str(self) -> &'static str {
        match self {
            TaskState::New => "New",
            TaskState::Ready => "Ready",
            TaskState::Running => "Running",
            TaskState::Blocked => "Blocked",
            TaskState::Exited => "Exited",
        }
    }
}

/// Canonical public representation of a schedulable execution unit.
///
/// Corresponds to the `thingos.task` schema kind (v2).  The kernel's internal
/// `Thread` structure is the current transitional backing; the `bridge` module
/// in `kernel::task` converts `Thread` state into this type.
///
/// ## Transitional notes
///
/// - `Thread` still backs Task execution internally; there is no first-class
///   kernel `Task` object yet.
/// - `Process` provisionally backs Job association; `job` carries the PID of
///   the owning process as a stand-in Job ID until the Job extraction is
///   complete.
/// - `name` is introspective/public metadata only — it is not yet a full
///   internal identity model.  Missing information is represented explicitly as
///   `None` rather than forcing a placeholder.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Task {
    /// The current lifecycle state of this task.
    pub state: TaskState,
    /// The Job (process) this task belongs to, if any.
    ///
    /// Provisionally carries the owning process's PID as a stand-in Job ID.
    /// `None` for kernel-only threads that have no associated process.
    pub job: Option<u32>,
    /// A human-readable name for this task, if one has been assigned.
    ///
    /// Sourced from the thread-level name store (`Thread::name`).  `None`
    /// when no name has been set via `set_current_task_name`.
    pub name: Option<alloc::string::String>,
}
