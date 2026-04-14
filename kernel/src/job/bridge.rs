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
//! * **Phase 9** (current): lifecycle state grouped into `ProcessLifecycle`
//!   as the extraction seam for `Job`.
//!
//! # Preferred source: `ProcessLifecycle`
//!
//! As of Phase 9, lifecycle-oriented state inside `Process` is grouped under
//! [`crate::task::ProcessLifecycle`] (`Process.lifecycle`).  This bridge is
//! the **canonical public surface** for that data; new code mapping lifecycle
//! state to `Job` / `JobExit` / `JobWaitResult` should read from
//! `Process.lifecycle` rather than from top-level `Process` fields.
//!
//! | `ProcessLifecycle` field     | Canonical `Job` concept          |
//! |------------------------------|----------------------------------|
//! | `lifecycle.ppid`             | Job parent/child linkage         |
//! | `lifecycle.thread_ids`       | Job thread-group membership      |
//! | `lifecycle.exec_in_progress` | Job lifecycle gate               |
//! | `lifecycle.children_done`    | Job wait queue                   |
//!
//! The Phase 9 integration is visible in `ProcessSnapshot::thread_states`:
//! `list_processes` now populates that field from `lifecycle.thread_ids`,
//! so `job_state_from_snapshot` accurately reflects the full thread-group
//! state rather than only the thread-group leader's state.
//!
//! # Entry points
//!
//! | Code context                              | Preferred function                   |
//! |-------------------------------------------|--------------------------------------|
//! | Has a `ProcessSnapshot` (e.g. procfs)     | `job_state_from_snapshot`            |
//! | Has `ProcessLifecycle` + thread states    | `job_state_from_lifecycle`           |
//! | Has raw `&[ThreadState]`                  | `job_state_from_thread_states`       |
//! | Mapping `poll_task_exit` result           | `job_wait_result_from_poll`          |
//! | Mapping `waitpid` result                  | `job_wait_result_from_waitpid`       |
//!
//! # Future direction
//!
//! Once `Process` is decomposed, this bridge will shrink.  The thread-count
//! heuristic used here will be replaced by an explicit lifecycle transition
//! embedded in the scheduler (spawned → running → exited).  The
//! `ProcessLifecycle` subdivision introduced in Phase 9 is the extraction seam
//! that will allow this bridge to read from a first-class `Job` object instead
//! of a `Process`-shaped snapshot.

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
/// Uses `snapshot.thread_states` when non-empty — this slice is populated from
/// [`crate::task::ProcessLifecycle::thread_ids`] in Phase 9 and gives accurate
/// `JobState` for multi-threaded processes.  Falls back to `[snapshot.state]`
/// (the thread-group leader's state alone) for legacy test helpers or code
/// paths that have not yet been updated.
pub fn job_state_from_snapshot(
    snapshot: &crate::sched::hooks::ProcessSnapshot,
) -> JobState {
    if !snapshot.thread_states.is_empty() {
        job_state_from_thread_states(&snapshot.thread_states)
    } else {
        // Fallback: only the group-leader state is available (e.g. test
        // helpers that don't populate thread_states).  Treat as a
        // single-thread process — correct for single-threaded processes and
        // conservative for multi-threaded ones (may undercount).
        job_state_from_thread_states(&[snapshot.state])
    }
}

// ── JobState from ProcessLifecycle (Phase 9) ─────────────────────────────────

/// Derive `JobState` from a [`crate::task::ProcessLifecycle`] and the current
/// states of its threads.
///
/// This is the **preferred** bridge entry point for code that already holds
/// (or can directly access) a `ProcessLifecycle` — e.g. code running under
/// the process mutex in the scheduler.  Callers should supply the live
/// `ThreadState` for each TID in `lifecycle.thread_ids`, in any order.
///
/// The `lifecycle` parameter is accepted for documentation clarity and
/// future use; today the mapping is derived entirely from `thread_states`.
/// Future phases will use `lifecycle` fields directly once `Job` is a
/// first-class kernel object.
///
/// # Relationship to `job_state_from_thread_states`
///
/// This is a thin, explicitly-named wrapper over
/// [`job_state_from_thread_states`] that makes the intended data-flow
/// (`ProcessLifecycle` → `Job`) visible in the call graph.  Prefer it over
/// calling `job_state_from_thread_states` directly when `ProcessLifecycle` is
/// in scope.
pub fn job_state_from_lifecycle(
    lifecycle: &crate::task::ProcessLifecycle,
    thread_states: &[ThreadState],
) -> JobState {
    // Debug sanity: the caller should provide at most as many states as there
    // are known TIDs in the lifecycle.  Extra states are harmless but indicate
    // a likely bug in the caller.
    debug_assert!(
        thread_states.len() <= lifecycle.thread_ids.len(),
        "thread_states has more entries ({}) than lifecycle.thread_ids ({})",
        thread_states.len(),
        lifecycle.thread_ids.len(),
    );
    job_state_from_thread_states(thread_states)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sched::hooks::ProcessSnapshot;
    use crate::task::TaskState;

    fn make_snapshot(state: TaskState, exit_code: Option<i32>) -> ProcessSnapshot {
        ProcessSnapshot {
            pid: 1,
            ppid: 0,
            tid: 1,
            name: alloc::string::String::from("test"),
            state,
            argv: alloc::vec::Vec::new(),
            exec_path: alloc::string::String::new(),
            exit_code,
            pgid: 1,
            sid: 1,
            session_leader: false,
            cwd: alloc::string::String::from("/"),
            namespace_label: alloc::string::String::from("global"),
            // Phase 9: populate thread_states so job_state_from_snapshot uses
            // the full slice path rather than the single-leader fallback.
            thread_states: alloc::vec![state],
        }
    }

    // ── job_state_from_thread_states ─────────────────────────────────────────

    #[test]
    fn test_job_state_empty_is_new() {
        assert_eq!(job_state_from_thread_states(&[]), JobState::New);
    }

    #[test]
    fn test_job_state_runnable_thread_is_running() {
        assert_eq!(
            job_state_from_thread_states(&[ThreadState::Runnable]),
            JobState::Running
        );
    }

    #[test]
    fn test_job_state_running_thread_is_running() {
        assert_eq!(
            job_state_from_thread_states(&[ThreadState::Running]),
            JobState::Running
        );
    }

    #[test]
    fn test_job_state_blocked_thread_is_running() {
        assert_eq!(
            job_state_from_thread_states(&[ThreadState::Blocked]),
            JobState::Running
        );
    }

    #[test]
    fn test_job_state_all_dead_is_exited() {
        assert_eq!(
            job_state_from_thread_states(&[ThreadState::Dead, ThreadState::Dead]),
            JobState::Exited
        );
    }

    #[test]
    fn test_job_state_mixed_live_and_dead_is_running() {
        assert_eq!(
            job_state_from_thread_states(&[ThreadState::Dead, ThreadState::Runnable]),
            JobState::Running
        );
    }

    // ── job_from_thread_states ───────────────────────────────────────────────

    #[test]
    fn test_job_from_thread_states_wraps_state() {
        let job = job_from_thread_states(&[ThreadState::Runnable]);
        assert_eq!(job.state, JobState::Running);
    }

    // ── job_exit_from_snapshot ───────────────────────────────────────────────

    #[test]
    fn test_job_exit_live_process_no_code() {
        let snap = make_snapshot(TaskState::Runnable, None);
        let exit = job_exit_from_snapshot(&snap);
        assert_eq!(exit.state, JobState::Running);
        assert_eq!(exit.code, None);
    }

    #[test]
    fn test_job_exit_dead_process_has_code() {
        let snap = make_snapshot(TaskState::Dead, Some(42));
        let exit = job_exit_from_snapshot(&snap);
        assert_eq!(exit.state, JobState::Exited);
        assert_eq!(exit.code, Some(42));
    }

    #[test]
    fn test_job_exit_live_process_stale_code_discarded() {
        // A running process that somehow has a stale exit_code should not leak
        // that code into the canonical JobExit representation.
        let snap = make_snapshot(TaskState::Running, Some(99));
        let exit = job_exit_from_snapshot(&snap);
        assert_eq!(exit.state, JobState::Running);
        assert_eq!(exit.code, None);
    }

    // ── job_wait_result_from_poll ────────────────────────────────────────────

    #[test]
    fn test_poll_none_is_running() {
        assert_eq!(job_wait_result_from_poll(None), JobWaitResult::Running);
    }

    #[test]
    fn test_poll_some_is_exited_with_code() {
        assert_eq!(
            job_wait_result_from_poll(Some(0)),
            JobWaitResult::Exited { code: Some(0) }
        );
        assert_eq!(
            job_wait_result_from_poll(Some(1)),
            JobWaitResult::Exited { code: Some(1) }
        );
        assert_eq!(
            job_wait_result_from_poll(Some(-1)),
            JobWaitResult::Exited { code: Some(-1) }
        );
    }

    // ── job_wait_result_from_waitpid ─────────────────────────────────────────

    #[test]
    fn test_waitpid_produces_exited_with_code() {
        assert_eq!(
            job_wait_result_from_waitpid(0),
            JobWaitResult::Exited { code: Some(0) }
        );
        assert_eq!(
            job_wait_result_from_waitpid(127),
            JobWaitResult::Exited { code: Some(127) }
        );
    }

    // ── job_state_from_snapshot (multi-thread, Phase 9) ──────────────────────

    fn make_snapshot_with_threads(
        leader_state: TaskState,
        all_thread_states: alloc::vec::Vec<TaskState>,
    ) -> ProcessSnapshot {
        ProcessSnapshot {
            pid: 1,
            ppid: 0,
            tid: 1,
            name: alloc::string::String::from("test"),
            state: leader_state,
            argv: alloc::vec::Vec::new(),
            exec_path: alloc::string::String::new(),
            exit_code: None,
            pgid: 1,
            sid: 1,
            session_leader: false,
            cwd: alloc::string::String::from("/"),
            namespace_label: alloc::string::String::from("global"),
            thread_states: all_thread_states,
        }
    }

    /// When the leader has exited but a sibling thread is still alive, the
    /// canonical JobState must be Running, not Exited.
    #[test]
    fn test_job_state_from_snapshot_uses_all_thread_states() {
        // Leader is Dead but a sibling is still Runnable.
        let snap = make_snapshot_with_threads(
            TaskState::Dead,
            alloc::vec![TaskState::Dead, TaskState::Runnable],
        );
        assert_eq!(
            job_state_from_snapshot(&snap),
            JobState::Running,
            "group is Running while any sibling thread is alive"
        );
    }

    /// When thread_states is empty the bridge falls back to the leader state.
    #[test]
    fn test_job_state_from_snapshot_fallback_when_thread_states_empty() {
        let snap = make_snapshot_with_threads(TaskState::Dead, alloc::vec::Vec::new());
        assert_eq!(
            job_state_from_snapshot(&snap),
            JobState::Exited,
            "empty thread_states should fall back to leader state"
        );
    }

    // ── job_state_from_lifecycle ─────────────────────────────────────────────

    #[test]
    fn test_job_state_from_lifecycle_all_dead_is_exited() {
        let lifecycle = crate::task::ProcessLifecycle::new(0, 1);
        assert_eq!(
            job_state_from_lifecycle(&lifecycle, &[ThreadState::Dead]),
            JobState::Exited
        );
    }

    #[test]
    fn test_job_state_from_lifecycle_live_thread_is_running() {
        let lifecycle = crate::task::ProcessLifecycle::new(0, 1);
        assert_eq!(
            job_state_from_lifecycle(&lifecycle, &[ThreadState::Runnable]),
            JobState::Running
        );
    }

    #[test]
    fn test_job_state_from_lifecycle_empty_is_new() {
        let lifecycle = crate::task::ProcessLifecycle::new(0, 1);
        assert_eq!(job_state_from_lifecycle(&lifecycle, &[]), JobState::New);
    }
}
