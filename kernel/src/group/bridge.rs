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
/// When a public TTY-foreground query becomes available this logic will be
/// replaced with an exact `foreground_pgid == pgid` comparison.
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
