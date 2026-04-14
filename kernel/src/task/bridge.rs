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
/// Bridge from the kernel's internal Thread/Process model to the canonical
/// Task vocabulary generated from `tools/kindc/kinds/task.kind`.
///
/// # Transitional mapping
///
/// Thing-OS is migrating from a Unix-derived Process/Thread model toward a
/// Job/Task model.  This module is the explicit, single-point bridge between
/// the two during the transition:
///
/// | Current internal type | Canonical target |
/// |-----------------------|-----------------|
/// | `Thread<R>`           | `Task`          |
/// | `ThreadState`         | `TaskState`     |
/// | `Process`             | `Job` (future)  |
///
/// Code that needs to expose task-shaped data at system boundaries (procfs,
/// syscall responses, debug output) should go through these conversions rather
/// than mapping ad-hoc.  When the internal scheduler eventually renames its
/// own structures, this bridge narrows the blast radius to a single file.
///
/// # State correspondence
///
/// ```text
/// ThreadState::Runnable  →  TaskState::Ready    (on run queue, waiting for CPU)
/// ThreadState::Running   →  TaskState::Running  (currently executing)
/// ThreadState::Blocked   →  TaskState::Blocked  (waiting for event)
/// ThreadState::Dead      →  TaskState::Exited   (has exited; exit code available)
/// ```
///
/// `TaskState::New` is not currently reachable through a live `ThreadState`
/// because threads are inserted into the registry in `Runnable` state.  It is
/// reserved for a future phase where the scheduler will emit a brief `New`
/// window between allocation and first enqueue.

use crate::generated::{Task, TaskState};
use crate::task::ThreadState;

/// Convert a kernel-internal `ThreadState` into the canonical `TaskState`.
///
/// See the module-level documentation for the full state correspondence table.
pub fn thread_state_to_task_state(state: ThreadState) -> TaskState {
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
/// Construct a canonical `Task` from a kernel-internal `ThreadState`.
///
/// This is the primary entry point for any code that needs to hand a
/// `Task`-shaped value to a public-facing boundary (procfs, syscall payload,
/// debug report, etc.).
pub fn thread_state_to_task(state: ThreadState) -> Task {
    Task {
        state: thread_state_to_task_state(state),
    }
}
