//! Bridge layer: kernel `Thread` → canonical `thingos::task::Task`.
//!
//! # Purpose
//!
//! This module is the **single conversion point** from the kernel's
//! transitional `Thread`-shaped execution model to the schema-generated
//! canonical `Task` representation.  No ad-hoc conversions should be
//! scattered elsewhere; all task-status-facing public paths go through here.
//!
//! # Transitional mapping
//!
//! | Kernel `ThreadState` | Canonical `TaskState` | Rationale                              |
//! |----------------------|-----------------------|----------------------------------------|
//! | (newly created)      | `New`                 | not yet enqueued — no direct mapping needed today |
//! | `Runnable`           | `Ready`               | eligible but not on CPU                |
//! | `Running`            | `Running`             | currently executing                    |
//! | `Blocked`            | `Blocked`             | waiting for event                      |
//! | `Dead`               | `Exited`              | has exited; exit code available        |
//!
//! The `New` state has no current kernel analogue because the kernel
//! creates threads already-runnable.  It is reserved for future use when
//! the scheduler gains an explicit "not yet started" state.
//!
//! # Future direction
//!
//! Once the internal `Thread` structure is renamed to `Task` this bridge
//! will shrink to a trivial identity mapping, and the canonical type will
//! be used directly.

use crate::sched::state::ThreadState;
use thingos::task::{Task, TaskState};

/// Convert a kernel `ThreadState` into the canonical `TaskState`.
///
/// This is the authoritative mapping; all scheduler-facing public paths that
/// need to report task state should call this function.
pub fn task_state_from_thread(state: ThreadState) -> TaskState {
    match state {
        ThreadState::Runnable => TaskState::Ready,
        ThreadState::Running => TaskState::Running,
        ThreadState::Blocked => TaskState::Blocked,
        ThreadState::Dead => TaskState::Exited,
    }
}

/// Build a canonical `Task` value from a kernel `ThreadState`.
///
/// Convenience wrapper around [`task_state_from_thread`] for callers that
/// need a full `Task` struct rather than just the state.
pub fn task_from_thread_state(state: ThreadState) -> Task {
    Task { state: task_state_from_thread(state) }
}
