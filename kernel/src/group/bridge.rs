//! Bridge layer: kernel coordination state → canonical `thingos::group::Group`.
//!
//! # Purpose
//!
//! This module is the **single conversion point** from the kernel's Unix-shaped
//! coordination structures (`Process::pgid`, `Process::sid`,
//! `Process::session_leader`, `ConsoleTtyState::foreground_pgid`) to the
//! schema-generated canonical `Group` and `GroupKind` types.
//!
//! All group-coordination-facing public paths (procfs, future session/TTY
//! boundary) should go through here rather than reading internal fields
//! directly.
//!
//! # Transitional mapping
//!
//! | Kernel source                               | Canonical `GroupKind`     |
//! |---------------------------------------------|---------------------------|
//! | `Process::session_leader == true`            | `Foreground`              |
//! | `foreground_pgid == Some(process.pgid)`      | `Foreground`              |
//! | otherwise                                   | `Coordination`            |
//!
//! The `foreground_pgid` check is the closer semantic match to terminal
//! foreground ownership, but because `ConsoleTtyState` is behind a global
//! `Mutex` that is not exposed outside `devfs`, the bridge currently falls
//! back to `session_leader` as a conservative proxy.  When the TTY foreground
//! query is made available from outside `devfs`, this bridge will be updated
//! to use it directly.
//!
//! # What Process/Unix concepts are not (yet) replaced
//!
//! * Signal broadcast domains (`pgid` → `send_signal_to_group`)
//! * `TIOCSPGRP` / `TIOCGPGRP` ioctl group control
//! * `setsid` / session creation
//! * `SIGCONT`, `SIGSTOP`, `SIGHUP` routing
//!
//! These remain tied to Unix-shaped internals.  They become `Group` in
//! Phase 5 and beyond.
//!
//! # Future direction
//!
//! Once a stable query interface for `ConsoleTtyState::foreground_pgid` is
//! available outside `devfs`, `group_kind_from_snapshot` will switch to the
//! direct TTY check.  When signal routing migrates into `Group`, this bridge
//! will gain a `group_from_process` constructor that carries membership.

use thingos::group::{Group, GroupKind};

/// Derive the canonical `GroupKind` from a process snapshot.
///
/// # Provisional heuristic
///
/// Phase 4 does not yet expose `ConsoleTtyState::foreground_pgid` outside
/// `devfs`, so this function uses `session_leader` as a conservative proxy:
///
/// * A session leader almost always starts as the foreground process, and
///   TTY-owning shells typically remain session leaders for the duration of
///   their life.
/// * Background processes and non-leader coordination groups report as
///   `Coordination`.
///
/// # Known limitations
///
/// This heuristic produces false positives and false negatives in several
/// real-world cases:
///
/// * **False positive**: A session leader that has moved itself to the
///   background (e.g., via `&` or `bg`) still reports `Foreground` here,
///   even though it no longer holds TTY foreground control.
/// * **False negative**: A non-leader process that has acquired foreground
///   control via `TIOCSPGRP` (e.g., a shell forked child that calls
///   `tcsetpgrp`) reports `Coordination` even though it is the actual TTY
///   foreground group.
/// * **Multi-process groups**: When the foreground pgid belongs to a group
///   whose leader is not a session leader, all members will report
///   `Coordination`.
///
/// When a public TTY-foreground query becomes available outside `devfs`
/// (i.e., `ConsoleTtyState::foreground_pgid` is accessible here), this
/// logic will be replaced with an exact `foreground_pgid == pgid` comparison
/// and the above corner cases will be eliminated.
pub fn group_kind_from_snapshot(
    snapshot: &crate::sched::hooks::ProcessSnapshot,
) -> GroupKind {
    if snapshot.session_leader {
        GroupKind::Foreground
    } else {
        GroupKind::Coordination
    }
}

/// Build a canonical `Group` from a process snapshot.
///
/// Convenience wrapper around [`group_kind_from_snapshot`].
pub fn group_from_snapshot(snapshot: &crate::sched::hooks::ProcessSnapshot) -> Group {
    Group { kind: group_kind_from_snapshot(snapshot) }
}

/// Return `true` when this process is in the foreground coordination group.
///
/// Derives foreground status entirely from [`group_kind_from_snapshot`] so
/// that all Group-bridge logic remains in one place.  Callers must not
/// inspect `pgid`, `sid`, or `session_leader` directly.
///
/// # Usage
///
/// This function backs the `/proc/<pid>/foreground_group` procfs path.
/// It is the canonical Group-vocabulary answer to "is this process in the
/// foreground group?" and replaces any direct TTY/pgid inspection for new
/// introspection surfaces.
pub fn foreground_group_from_snapshot(
    snapshot: &crate::sched::hooks::ProcessSnapshot,
) -> bool {
    group_kind_from_snapshot(snapshot) == GroupKind::Foreground
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sched::hooks::ProcessSnapshot;
    use crate::task::TaskState;

    fn make_snapshot(session_leader: bool) -> ProcessSnapshot {
        ProcessSnapshot {
            pid: 1,
            ppid: 0,
            tid: 1,
            name: alloc::string::String::from("test"),
            state: TaskState::Runnable,
            argv: alloc::vec::Vec::new(),
            exec_path: alloc::string::String::new(),
            exit_code: None,
            pgid: 1,
            sid: 1,
            session_leader,
            cwd: alloc::string::String::from("/"),
            namespace_label: alloc::string::String::from("global"),
        }
    }

    // ── group_kind_from_snapshot ─────────────────────────────────────────────

    #[test]
    fn test_session_leader_is_foreground() {
        let snap = make_snapshot(true);
        assert_eq!(group_kind_from_snapshot(&snap), GroupKind::Foreground);
    }

    #[test]
    fn test_non_leader_is_coordination() {
        let snap = make_snapshot(false);
        assert_eq!(group_kind_from_snapshot(&snap), GroupKind::Coordination);
    }

    // ── foreground_group_from_snapshot ──────────────────────────────────────

    #[test]
    fn test_foreground_group_from_snapshot_session_leader() {
        let snap = make_snapshot(true);
        assert!(foreground_group_from_snapshot(&snap));
    }

    #[test]
    fn test_foreground_group_from_snapshot_non_leader() {
        let snap = make_snapshot(false);
        assert!(!foreground_group_from_snapshot(&snap));
    }

    // ── group_from_snapshot ──────────────────────────────────────────────────

    #[test]
    fn test_group_from_snapshot_session_leader() {
        let snap = make_snapshot(true);
        let group = group_from_snapshot(&snap);
        assert_eq!(group.kind, GroupKind::Foreground);
    }

    #[test]
    fn test_group_from_snapshot_non_leader() {
        let snap = make_snapshot(false);
        let group = group_from_snapshot(&snap);
        assert_eq!(group.kind, GroupKind::Coordination);
    }

    #[test]
    fn test_group_from_snapshot_as_text_foreground() {
        let snap = make_snapshot(true);
        let group = group_from_snapshot(&snap);
        let text = group.as_text();
        assert!(text.contains("kind: Foreground"), "unexpected: {}", text);
    }

    #[test]
    fn test_group_from_snapshot_as_text_coordination() {
        let snap = make_snapshot(false);
        let group = group_from_snapshot(&snap);
        let text = group.as_text();
        assert!(text.contains("kind: Coordination"), "unexpected: {}", text);
    }
}
