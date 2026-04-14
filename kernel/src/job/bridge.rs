//! Bridge layer: kernel `Process` snapshot → canonical `thingos::job::Job`.
//!
//! # Purpose
//!
//! This module is the **single conversion point** from the kernel's
//! transitional `Process`-shaped lifecycle model to the schema-generated
//! canonical `Job` representation.  All job-lifecycle-facing public paths
//! (procfs, introspection, status reporting) go through here.
//!
//! # Transitional mapping
//!
//! Job lifecycle state is derived from the observed `ThreadState` of the
//! threads belonging to the process, since `Process` itself does not carry
//! an explicit lifecycle state field in the current implementation.
//!
//! | Observed thread group state                   | Canonical `JobState` |
//! |-----------------------------------------------|----------------------|
//! | No threads known yet / process embryonic       | `New`                |
//! | At least one thread is Runnable/Running/Blocked | `Running`           |
//! | All known threads are Dead                    | `Exited`             |
//!
//! # What `Process` is not (yet)
//!
//! The current `Process` struct carries address-space ownership, FD tables,
//! namespace state, signal dispositions, and more.  None of those
//! responsibilities are part of `Job` today.  Future phases will migrate
//! those concerns to their own canonical kinds:
//!
//! * **Phase 3**: wait/exit reporting migrates into `Job` terms.
//! * **Phase 4**: introduce `Group` as the coordination truth.
//! * **Phase 5**: begin internal `Process` decomposition by responsibility.
//!
//! # Future direction
//!
//! Once `Process` is decomposed, this bridge will shrink.  The thread-count
//! heuristic used here will be replaced by an explicit lifecycle transition
//! embedded in the scheduler (spawned → running → exited).

use crate::sched::state::ThreadState;
use thingos::job::{Job, JobState};

/// Derive the canonical `JobState` from the lifecycle of a process's threads.
///
/// `thread_states` should contain the `ThreadState` of every thread that
/// belongs to the process (the thread-group leader and any additional
/// threads).  Pass an empty slice if no threads are known yet.
pub fn job_state_from_thread_states(thread_states: &[ThreadState]) -> JobState {
    if thread_states.is_empty() {
        return JobState::New;
    }

    let all_dead = thread_states.iter().all(|s| *s == ThreadState::Dead);
    if all_dead {
        JobState::Exited
    } else {
        JobState::Running
    }
}

/// Build a canonical `Job` value from the lifecycle of a process's threads.
///
/// Convenience wrapper around [`job_state_from_thread_states`].
pub fn job_from_thread_states(thread_states: &[ThreadState]) -> Job {
    Job { state: job_state_from_thread_states(thread_states) }
}

/// Derive `JobState` from a [`crate::sched::hooks::ProcessSnapshot`].
///
/// Because `ProcessSnapshot` carries only the primary thread's state today,
/// this mapping is conservative: a single dead thread-group leader is treated
/// as `Exited`.  Once the snapshot includes all TIDs this will be refined.
pub fn job_state_from_snapshot(
    snapshot: &crate::sched::hooks::ProcessSnapshot,
) -> JobState {
    // Single-element slice is intentional: ProcessSnapshot today carries only
    // the thread-group leader's state.  When ProcessSnapshot is extended to
    // include all TIDs this call site will pass the full slice.
    job_state_from_thread_states(&[snapshot.state])
}
