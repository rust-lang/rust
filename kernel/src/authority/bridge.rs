//! Bridge layer: kernel `ProcessSnapshot` → canonical `thingos::authority::Authority`.
//!
//! # Purpose
//!
//! This module is the **single conversion point** from the kernel's
//! transitional `Process`-shaped permission context to the schema-generated
//! canonical `Authority` representation.  All authority-facing public paths
//! (procfs `/proc/<pid>/authority`, future introspection syscalls) should go
//! through here rather than reading internal fields directly.
//!
//! # Migration inventory — Process credential/permission responsibilities
//!
//! The table below inventories every field in the current `Process` struct that
//! answers authorization questions, and maps it to its intended future
//! `Authority` field.  Fields marked **provisional** remain in `Process`
//! pending fuller extraction.
//!
//! | `Process` / `ProcessSnapshot` field | Authorization role               | Intended `Authority` mapping          | Status       |
//! |--------------------------------------|----------------------------------|---------------------------------------|--------------|
//! | `name` (thread/process name)         | Human-readable identity label    | `Authority::name`                     | **Bridged**  |
//! | `exec_path`                          | Executable identity              | `Authority::name` (fallback if empty) | Provisional  |
//! | `pid` / `ppid`                       | Process identity                 | Future principal identifier           | Provisional  |
//! | `pgid` / `sid`                       | Coordination group identity      | `Group` domain (Phase 4)              | Provisional  |
//! | `session_leader`                     | TTY foreground ownership         | `Group::kind` (Phase 4)               | Provisional  |
//! | *(no uid/gid field yet)*             | POSIX user/group identity        | Future `Authority` fields             | Not yet added|
//! | *(no capability mask yet)*           | Fine-grained privilege           | `Authority::capabilities`             | Not yet added|
//! | `fd_table` (open files)              | Resource access rights           | Out of `Authority` scope (Phase 8+)   | Provisional  |
//! | `namespace`                          | VFS visibility                   | Future `Place` context (Phase 8)      | Provisional  |
//! | `signals` (signal dispositions)      | Signal delivery permissions      | Future authority concern (Phase 9+)   | Provisional  |
//!
//! # Transitional mapping
//!
//! | `ProcessSnapshot` field | `Authority` field     | Notes                                      |
//! |-------------------------|-----------------------|--------------------------------------------|
//! | `name`                  | `name`                | Thread/process name used as authority label|
//! | *(none)*                | `capabilities`        | Empty in Phase 7; no capability field yet  |
//!
//! # What is not yet replaced
//!
//! * uid/gid-like identity — no such field in `Process` yet
//! * capability masks — no such field in `Process` yet
//! * service-account / principal binding — not yet introduced
//! * signal permissions — remain in `Process::signals` (provisional)
//! * namespace / VFS visibility — will become `Place` context in Phase 8
//!
//! # Future direction
//!
//! When a uid/gid field or capability mask is added to `Process`, this bridge
//! will be the **only** place that needs updating to surface those fields
//! through the canonical `Authority` type.  New access-control code must not
//! read those fields from `Process` directly; all public paths must go through
//! this bridge.

use thingos::authority::Authority;

/// Build a canonical `Authority` from a [`crate::sched::hooks::ProcessSnapshot`].
///
/// # Transitional mapping
///
/// In Phase 7 the authority `name` is taken from `snapshot.name` (the
/// thread/process name).  When `name` is empty, `exec_path` is used as the
/// fallback so the authority is always non-empty and identifiable.
///
/// `capabilities` is always empty in Phase 7 because `Process` carries no
/// explicit capability mask.  Once a capability field is added to `Process`
/// this function will be the sole site that reads and surfaces it.
///
/// # Note on provisional credential state
///
/// The current `Process` struct does not carry uid/gid, capability masks, or
/// service-account fields.  All credential/permission state in `Process` is
/// therefore **provisional** — it backs the canonical `Authority` through this
/// bridge but has not yet been fully extracted into `Authority`-shaped storage.
pub fn authority_from_snapshot(
    snapshot: &crate::sched::hooks::ProcessSnapshot,
) -> Authority {
    // Prefer the human-readable thread/process name; fall back to exec_path when
    // the name has not been set (i.e., it is empty).
    //
    // PROVISIONAL: This derives the authority name from the current process/thread
    // name.  Future phases will replace this with a stable principal identifier
    // once uid/gid-like fields or a service-account concept are introduced into
    // the Process struct.
    let name = if snapshot.name.is_empty() {
        snapshot.exec_path.clone()
    } else {
        snapshot.name.clone()
    };

    // PROVISIONAL: capabilities is always empty in Phase 7.  The current
    // `Process` struct carries no capability mask.  When a capability field is
    // added to `Process` (or its Authority-shaped substructure from Phase 5),
    // this is the sole site that must be updated to surface those capabilities
    // through the canonical `Authority` type.  New access-control code must NOT
    // read capability state from `Process` directly.
    Authority {
        name,
        capabilities: alloc::vec::Vec::new(),
    }
}
