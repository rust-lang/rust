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
//!
//! # Field optionality
//!
//! The `job` field in the canonical `Task` carries the owning process's PID
//! as a provisional Job ID.  It is `None` for kernel-only threads that have no
//! associated `Process`.
//!
//! The `name` field is sourced from the thread-level name store.  It is `None`
//! when no human-readable label has been assigned.
//!
//! # Transitional mapping table (v2)
//!
//! | Canonical `Task` field | Current kernel source                    |
//! |------------------------|------------------------------------------|
//! | `state`                | `ThreadState` (via `task_state_from_thread`) |
//! | `job`                  | `Process::pid` (provisional Job ID)      |
//! | `name`                 | `Thread::name` / `ProcessSnapshot::name` |
//!
//! Both `Thread` and `Process` remain the transitional backing; first-class
//! `Task` and `Job` kernel objects will replace them as the scheduler migrates.

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
/// This is a minimal bridge that populates only `state`; `job` and `name` are
/// left as `None`.  Prefer [`task_from_snapshot`] when a `ProcessSnapshot` is
/// available so that the richer Task shape can be fully populated.
pub fn task_from_thread_state(state: ThreadState) -> Task {
    Task {
        state: task_state_from_thread(state),
        job: None,
        name: None,
    }
}

/// Construct a canonical `Task` from a [`crate::sched::hooks::ProcessSnapshot`].
///
/// Populates all three canonical fields:
///
/// * `state` — derived from `snapshot.state` via `task_state_from_thread`.
/// * `job`   — set to `Some(snapshot.pid)` as the provisional Job ID (Process
///             PID).  This reflects the current `Process`-as-Job transitional
///             mapping; it will be replaced by a first-class Job ID once Job
///             extraction is complete.
/// * `name`  — set to `Some(snapshot.name)` if the name is non-empty, else
///             `None`.  Sourced from the thread-level name store visible in
///             `ProcessSnapshot::name`.
///
/// This function is the preferred entry point for any code that already holds
/// a `ProcessSnapshot` and needs a fully-populated canonical `Task`.
pub fn task_from_snapshot(snapshot: &crate::sched::hooks::ProcessSnapshot) -> Task {
    let state = task_state_from_thread(snapshot.state);
    let job = Some(snapshot.pid);
    let name = (!snapshot.name.is_empty()).then(|| snapshot.name.clone());
    Task { state, job, name }
}

// --- Compatibility Aliases ---

/// Alias for [`task_state_from_thread`].
pub use task_state_from_thread as thread_state_to_task_state;

/// Alias for [`task_from_thread_state`].
pub use task_from_thread_state as thread_state_to_task;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sched::hooks::ProcessSnapshot;

    fn make_snapshot(state: ThreadState, name: &str, pid: u32) -> ProcessSnapshot {
        ProcessSnapshot {
            pid,
            ppid: 0,
            tid: pid as u64,
            name: alloc::string::String::from(name),
            state,
            argv: alloc::vec::Vec::new(),
            exec_path: alloc::string::String::new(),
            exit_code: None,
            pgid: pid,
            sid: pid,
            session_leader: false,
            cwd: alloc::string::String::from("/"),
            namespace_label: alloc::string::String::from("global"),
            thread_states: alloc::vec![state],
        }
    }

    // ── task_state_from_thread ───────────────────────────────────────────────

    #[test]
    fn test_runnable_maps_to_ready() {
        assert_eq!(task_state_from_thread(ThreadState::Runnable), TaskState::Ready);
    }

    #[test]
    fn test_running_maps_to_running() {
        assert_eq!(task_state_from_thread(ThreadState::Running), TaskState::Running);
    }

    #[test]
    fn test_blocked_maps_to_blocked() {
        assert_eq!(task_state_from_thread(ThreadState::Blocked), TaskState::Blocked);
    }

    #[test]
    fn test_dead_maps_to_exited() {
        assert_eq!(task_state_from_thread(ThreadState::Dead), TaskState::Exited);
    }

    // ── task_from_thread_state ───────────────────────────────────────────────

    #[test]
    fn test_task_from_thread_state_has_correct_state() {
        let task = task_from_thread_state(ThreadState::Runnable);
        assert_eq!(task.state, TaskState::Ready);
    }

    #[test]
    fn test_task_from_thread_state_has_no_job() {
        let task = task_from_thread_state(ThreadState::Running);
        assert_eq!(task.job, None);
    }

    #[test]
    fn test_task_from_thread_state_has_no_name() {
        let task = task_from_thread_state(ThreadState::Blocked);
        assert_eq!(task.name, None);
    }

    // ── task_from_snapshot ───────────────────────────────────────────────────

    #[test]
    fn test_task_from_snapshot_state_is_mapped() {
        let snap = make_snapshot(ThreadState::Runnable, "proc", 1);
        let task = task_from_snapshot(&snap);
        assert_eq!(task.state, TaskState::Ready);
    }

    #[test]
    fn test_task_from_snapshot_job_is_pid() {
        let snap = make_snapshot(ThreadState::Running, "proc", 42);
        let task = task_from_snapshot(&snap);
        assert_eq!(task.job, Some(42));
    }

    #[test]
    fn test_task_from_snapshot_name_is_set_when_non_empty() {
        let snap = make_snapshot(ThreadState::Running, "my-proc", 1);
        let task = task_from_snapshot(&snap);
        assert_eq!(task.name.as_deref(), Some("my-proc"));
    }

    #[test]
    fn test_task_from_snapshot_name_is_none_when_empty() {
        let snap = make_snapshot(ThreadState::Running, "", 1);
        let task = task_from_snapshot(&snap);
        assert_eq!(task.name, None);
    }

    #[test]
    fn test_task_from_snapshot_dead_state_is_exited() {
        let snap = make_snapshot(ThreadState::Dead, "zombie", 7);
        let task = task_from_snapshot(&snap);
        assert_eq!(task.state, TaskState::Exited);
    }

    // ── alias coverage ───────────────────────────────────────────────────────

    #[test]
    fn test_thread_state_to_task_state_alias() {
        assert_eq!(
            thread_state_to_task_state(ThreadState::Dead),
            TaskState::Exited
        );
    }

    #[test]
    fn test_thread_state_to_task_alias() {
        let task = thread_state_to_task(ThreadState::Blocked);
        assert_eq!(task.state, TaskState::Blocked);
        assert_eq!(task.job, None);
        assert_eq!(task.name, None);
    }
}
