//! Bridge layer: kernel `Process` snapshot → canonical `thingos::job` types.
//!
//! # Purpose
//!
//! This module is the **single conversion point** from the kernel's
//! transitional `Process`-shaped lifecycle model to the schema-generated
//! canonical `Job`, `JobExit`, and `JobWaitResult` representations.  All
//! job-lifecycle-facing public paths (procfs, introspection, status reporting)
//! go through here.
//!
//! # Transitional mapping
//!
//! Job lifecycle state is derived from the observed `ThreadState` of the
//! threads belonging to the process, since `Process` itself does not carry
//! an explicit lifecycle state field in the current implementation.
//!
//! | Observed thread group state                    | Canonical `JobState`  |
//! |------------------------------------------------|-----------------------|
//! | No threads known yet / process embryonic        | `New`                 |
//! | At least one thread is Runnable/Running/Blocked | `Running`             |
//! | All known threads are Dead                     | `Exited`              |
//!
//! Exit and wait results (Phase 3):
//!
//! | Kernel source                        | Canonical type          | Value                            |
//! |--------------------------------------|-------------------------|----------------------------------|
//! | `ProcessSnapshot` (state + exit_code)| `JobExit`               | state + code forwarded           |
//! | `poll_task_exit` → `None`            | `JobWaitResult::Running`|                                  |
//! | `poll_task_exit` → `Some(code)`      | `JobWaitResult::Exited` | `code: Some(code)`               |
//! | `waitpid` → `(_, code)`              | `JobWaitResult::Exited` | `code: Some(code)`               |
//!
//! # What `Process` is not (yet)
//!
//! The current `Process` struct carries address-space ownership, FD tables,
//! namespace state, signal dispositions, and more.  None of those
//! responsibilities are part of `Job` today.  Future phases will migrate
//! those concerns to their own canonical kinds:
//!
//! * **Phase 3** (this phase): wait/exit reporting added to the `Job` bridge.
//! * **Phase 4**: introduce `Group` as the coordination truth.
//! * **Phase 5**: begin internal `Process` decomposition by responsibility.
//!
//! # Future direction
//!
//! Once `Process` is decomposed, this bridge will shrink.  The thread-count
//! heuristic used here will be replaced by an explicit lifecycle transition
//! embedded in the scheduler (spawned → running → exited).

use crate::sched::state::ThreadState;
use thingos::job::{Job, JobExit, JobState, JobWaitResult};

// ── JobState / Job ────────────────────────────────────────────────────────────

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

// ── JobExit (Phase 3) ─────────────────────────────────────────────────────────

/// Build a canonical `JobExit` from a [`crate::sched::hooks::ProcessSnapshot`].
///
/// The snapshot carries the thread-group leader's `ThreadState` and the
/// exit code stored when `mark_task_exited` was called.  When the leader is
/// still alive `code` is `None`.
///
/// # Transitional mapping
///
/// | Snapshot field      | `JobExit` field |
/// |---------------------|-----------------|
/// | `snapshot.state`    | `state` (via `job_state_from_snapshot`) |
/// | `snapshot.exit_code`| `code`          |
pub fn job_exit_from_snapshot(
    snapshot: &crate::sched::hooks::ProcessSnapshot,
) -> JobExit {
    let state = job_state_from_snapshot(snapshot);
    // Only forward exit_code when the job has actually exited.  The kernel may
    // still carry a stale exit_code field on a thread that has been reused or
    // whose state has not yet propagated; reporting it for a Running job would
    // be misleading.  We intentionally discard any value present while the job
    // is still live.
    let code = if state == JobState::Exited { snapshot.exit_code } else { None };
    JobExit { state, code }
}

// ── JobWaitResult (Phase 3) ───────────────────────────────────────────────────

/// Convert the result of `poll_task_exit` into a canonical `JobWaitResult`.
///
/// `poll_result` is the `Option<i32>` returned by
/// `crate::sched::poll_task_exit_current`:
///
/// * `None`       → the task is still alive → `JobWaitResult::Running`
/// * `Some(code)` → the task has exited     → `JobWaitResult::Exited { code: Some(code) }`
///
/// This is the **authoritative** mapping from the current poll machinery into
/// the canonical wait-result vocabulary.  Call sites in procfs and any future
/// wait syscall wrapper should use this function rather than their own ad-hoc
/// match.
pub fn job_wait_result_from_poll(poll_result: Option<i32>) -> JobWaitResult {
    match poll_result {
        None => JobWaitResult::Running,
        Some(code) => JobWaitResult::Exited { code: Some(code) },
    }
}

/// Convert the result of `waitpid` into a canonical `JobWaitResult`.
///
/// `waitpid` returns `(child_pid, exit_code)` on success.  This function
/// maps that to `JobWaitResult::Exited`.  The caller is responsible for
/// the `WNOHANG`/no-children distinction; pass only a confirmed exit here.
pub fn job_wait_result_from_waitpid(exit_code: i32) -> JobWaitResult {
    JobWaitResult::Exited { code: Some(exit_code) }
}
