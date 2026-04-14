//! Bridge layer: kernel-internal `ThreadState` → canonical `thingos::task::Task`.
//!
//! # Purpose
//!
//! This module is the **single conversion point** from the kernel's
//! transitional `Thread`-shaped execution model to the schema-generated
//! canonical `Task` representation.
//!
//! Thing-OS is migrating from a Unix-derived Process/Thread model toward a
//! Job/Task model. This bridge narrow the blast radius of changes to internal
//! machinery.
//!
//! # State correspondence
//!
//! | Kernel `ThreadState` | Canonical `TaskState` | Rationale                              |
//! |----------------------|-----------------------|----------------------------------------|
//! | `Runnable`           | `Ready`               | Eligible but not on CPU                |
//! | `Running`            | `Running`             | Currently executing                    |
//! | `Blocked`            | `Blocked`             | Waiting for event (I/O, lock, etc.)    |
//! | `Dead`               | `Exited`              | Has exited; exit code available        |
//!
//! `TaskState::New` is not currently reachable through a live `ThreadState`
//! because threads are currently inserted into the scheduler in `Runnable` state.

use crate::task::ThreadState;
use thingos::task::{Task, TaskState};

/// Convert a kernel-internal `ThreadState` into the canonical `TaskState`.
///
/// This is the authoritative mapping; all scheduler-facing public paths that
/// need to report task state (e.g. procfs, syscalls) should call this function.
pub fn task_state_from_thread(state: ThreadState) -> TaskState {
    match state {
        ThreadState::Runnable => TaskState::Ready,
        ThreadState::Running => TaskState::Running,
        ThreadState::Blocked => TaskState::Blocked,
        ThreadState::Dead => TaskState::Exited,
    }
}

/// Construct a canonical `Task` from a kernel-internal `ThreadState`.
///
/// This is the primary entry point for any code that needs to hand a
/// `Task`-shaped value to a public-facing boundary.
pub fn task_from_thread_state(state: ThreadState) -> Task {
    Task {
        state: task_state_from_thread(state),
    }
}

// --- Compatibility Aliases ---

/// Alias for [`task_state_from_thread`].
pub use task_state_from_thread as thread_state_to_task_state;

/// Alias for [`task_from_thread_state`].
pub use task_from_thread_state as thread_state_to_task;
