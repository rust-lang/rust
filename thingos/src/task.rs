//! Canonical public types for the `thingos.task` schema kind.
//!
//! # Schema (v1)
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
//! | Canonical field | Current kernel source          |
//! |-----------------|-------------------------------|
//! | `state`         | `Thread::state` (ThreadState) |
//!
//! `ThreadState::Runnable` maps to `TaskState::Ready` because the kernel
//! distinguishes "eligible to run" from "currently executing", which aligns
//! with the canonical `Ready`/`Running` split.

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
/// Corresponds to the `thingos.task` schema kind (v1).  The kernel's internal
/// `Thread` structure is the current transitional backing; the `bridge` module
/// in `kernel::task` converts `Thread` state into this type.
///
/// This struct is intentionally minimal for v1.  Additional fields (priority,
/// affinity, cpu, name) will be added as the bridge matures and the scheduler
/// migrates toward the Task/Job model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Task {
    /// The current lifecycle state of this task.
    pub state: TaskState,
}
