//! Bridge layer: kernel `ProcessSnapshot` → canonical `thingos::place::Place`.
//!
//! # Purpose
//!
//! This module is the **single conversion point** from the kernel's
//! transitional `Process`-shaped world/context model to the schema-generated
//! canonical `Place` representation.  All place-facing public paths
//! (procfs `/proc/<pid>/place`, future introspection syscalls) should go
//! through here rather than reading internal fields directly.
//!
//! # Migration inventory — Process world/context responsibilities
//!
//! | `Process` / `ProcessSnapshot` field | World-context role            | Intended `Place` mapping      | Status       |
//! |--------------------------------------|-------------------------------|-------------------------------|--------------|
//! | `cwd`                                | Current working directory     | `Place::cwd`                  | **Bridged**  |
//! | `namespace` (unit struct)            | VFS mount-table view          | `Place::namespace` (label)    | Provisional  |
//! | *(no root field yet)*                | Effective filesystem root     | `Place::root`                 | Not yet added|
//! | `env`                                | Inherited Unix env blob       | Legacy compat (quarantined)   | Provisional  |
//! | `argv` / `auxv`                      | Spawn-time invocation context | Legacy compat (quarantined)   | Provisional  |
//! | `pgid` / `sid` / `session_leader`   | Unix session/process-group    | `Group` domain (Phase 4/5)    | Provisional  |
//! | `fd_table`                           | Open-file resource table      | Future resource authority     | Provisional  |
//! | `signals`                            | Per-process signal state      | Future authority concern      | Provisional  |
//!
//! # Transitional mapping
//!
//! | `ProcessSnapshot` field  | `Place` field     | Notes                                           |
//! |--------------------------|-------------------|-------------------------------------------------|
//! | `cwd`                    | `cwd`             | Direct copy of the working directory path       |
//! | `namespace_label`        | `namespace`       | Always `"global"` in Phase 8 (unit NamespaceRef)|
//! | *(none)*                 | `root`            | Hardcoded `"/"` — no per-process root yet       |
//!
//! # What is not yet replaced
//!
//! * Per-process namespace isolation — `NamespaceRef` is a unit struct today;
//!   `Place::namespace` is always `"global"`.
//! * Per-process chroot / pivot-root — `Place::root` is always `"/"`.
//! * Inherited Unix environment blob (`Process::env`) — quarantined as legacy
//!   compatibility, not surfaced through Place.
//! * Terminal / UI attachment — belongs to `Presence` (not yet introduced).
//!
//! # Note on Presence
//!
//! **Presence has not yet been introduced as a live execution/interaction
//! concept.**  This bridge (Phase 8) surfaces only world-context fields.
//! Terminal attachment, UI/console attachment, and person-in-place
//! relationships will be added in a future phase under the `Presence` type.
//! New code must not conflate world-context (Place) with person-in-place
//! (Presence).
//!
//! # Future direction
//!
//! When per-process namespace isolation or chroot is added to `Process`, this
//! bridge will be the **only** place that needs updating to surface those
//! fields through the canonical `Place` type.  New world-context code must not
//! read cwd/namespace/root fields from `Process` directly; all public paths
//! must go through this bridge.

use thingos::place::Place;

/// Return the canonical [`Place`] for the **currently running task**.
///
/// This is the **preferred entry point** for any new world-context inspection
/// that needs to know "in what world is this execution occurring?".  Callers
/// should use this function instead of reaching into `Process` fields directly.
///
/// # Transitional behaviour
///
/// In Phase 8 this derives the `Place` from the current process's `cwd` field
/// (via `process_info_current`) and hard-codes `"global"` for the namespace
/// and `"/"` for the root.  When no process context is available (kernel
/// threads), `cwd` falls back to `"/"`.
///
/// # Migration note
///
/// Once per-process namespace isolation and chroot are added to `Process`,
/// this function remains the **single entry point** — callers will
/// transparently receive a richer `Place` without code changes at call sites.
pub fn place_for_current() -> Place {
    // PROVISIONAL: cwd is taken from ProcessInfo::cwd (the Process struct).
    // Future phases will replace this raw path with a stable VFS-node reference
    // once cwd tracking migrates into a Place-shaped substructure.
    let cwd = crate::sched::process_info_current()
        .map(|p| p.lock().cwd.clone())
        .unwrap_or_default();

    let cwd = if cwd.is_empty() {
        alloc::string::String::from("/")
    } else {
        cwd
    };

    // PROVISIONAL: namespace is always "global" in Phase 8 — all processes
    // share one global mount table.  Future phases will derive a stable
    // per-process namespace identifier from ProcessInfo::namespace.
    let namespace = alloc::string::String::from("global");

    // PROVISIONAL: root is always "/" because per-process chroot / pivot-root
    // is not yet implemented.
    let root = alloc::string::String::from("/");

    Place { cwd, namespace, root }
}

/// Build a canonical `Place` from a [`crate::sched::hooks::ProcessSnapshot`].
///
/// # Transitional mapping
///
/// In Phase 8:
/// * `cwd` is taken directly from `snapshot.cwd`.
/// * `namespace` is always `"global"` because `NamespaceRef` is a unit struct
///   and all processes share the same global mount table.
/// * `root` is always `"/"` because per-process chroot is not yet implemented.
///
/// # Note on provisional world-context state
///
/// The current `Process` struct does not carry an explicit per-process
/// namespace handle or chroot root.  All world-context state in `Process` is
/// therefore **provisional** — it backs the canonical `Place` through this
/// bridge but has not yet been fully extracted into `Place`-shaped storage.
///
/// # Note on Presence
///
/// Terminal attachment and UI/console attachment are **not** represented in
/// `Place`.  Those belong to `Presence`, which has not yet been introduced.
/// This function deliberately omits any terminal/session/console fields from
/// `ProcessSnapshot` even if they were available.
pub fn place_from_snapshot(
    snapshot: &crate::sched::hooks::ProcessSnapshot,
) -> Place {
    // PROVISIONAL: cwd is taken directly from Process::cwd.  Future phases
    // will replace this raw path string with a stable VFS-node reference once
    // cwd tracking migrates out of Process into a Place-shaped substructure.
    let cwd = if snapshot.cwd.is_empty() {
        alloc::string::String::from("/")
    } else {
        snapshot.cwd.clone()
    };

    // PROVISIONAL: namespace is always "global" in Phase 8.  Process::namespace
    // is a NamespaceRef unit struct — all processes share the same global mount
    // table.  When per-process namespace isolation is implemented, this will be
    // replaced with a stable namespace identifier derived from the per-process
    // NamespaceRef.
    let namespace = snapshot.namespace_label.clone();

    // PROVISIONAL: root is always "/" because per-process chroot / pivot-root
    // is not yet implemented.  When a per-process root binding is introduced
    // into Process, this bridge will be the sole site that reads and surfaces it.
    let root = alloc::string::String::from("/");

    Place { cwd, namespace, root }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sched::hooks::ProcessSnapshot;
    use crate::task::TaskState;

    fn make_snapshot(cwd: &str, namespace_label: &str) -> ProcessSnapshot {
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
            session_leader: false,
            cwd: alloc::string::String::from(cwd),
            namespace_label: alloc::string::String::from(namespace_label),
            thread_states: alloc::vec![TaskState::Runnable],
            space_id: thingos::space::SpaceId::NONE,
            space_mapping_count: 0,
            space_sharing_count: 0,
        }
    }

    // ── cwd mapping ──────────────────────────────────────────────────────────

    #[test]
    fn test_cwd_is_propagated_from_snapshot() {
        let snap = make_snapshot("/home/user", "global");
        let place = place_from_snapshot(&snap);
        assert_eq!(place.cwd, "/home/user");
    }

    #[test]
    fn test_empty_cwd_defaults_to_root() {
        let snap = make_snapshot("", "global");
        let place = place_from_snapshot(&snap);
        assert_eq!(place.cwd, "/");
    }

    // ── namespace mapping ─────────────────────────────────────────────────────

    #[test]
    fn test_namespace_label_is_propagated() {
        let snap = make_snapshot("/", "global");
        let place = place_from_snapshot(&snap);
        assert_eq!(place.namespace, "global");
    }

    // ── root mapping ──────────────────────────────────────────────────────────

    #[test]
    fn test_root_is_always_slash_in_phase_8() {
        let snap = make_snapshot("/work", "global");
        let place = place_from_snapshot(&snap);
        assert_eq!(place.root, "/");
    }

    // ── as_text output ────────────────────────────────────────────────────────

    #[test]
    fn test_place_from_snapshot_as_text_contains_cwd() {
        let snap = make_snapshot("/srv", "global");
        let place = place_from_snapshot(&snap);
        let text = place.as_text();
        assert!(text.contains("cwd: /srv"), "unexpected: {text}");
    }

    #[test]
    fn test_place_from_snapshot_as_text_contains_namespace() {
        let snap = make_snapshot("/", "global");
        let place = place_from_snapshot(&snap);
        let text = place.as_text();
        assert!(text.contains("namespace: global"), "unexpected: {text}");
    }

    #[test]
    fn test_place_from_snapshot_as_text_contains_root() {
        let snap = make_snapshot("/", "global");
        let place = place_from_snapshot(&snap);
        let text = place.as_text();
        assert!(text.contains("root: /"), "unexpected: {text}");
    }

    // ── place_for_current (no scheduler context) ──────────────────────────────

    #[test]
    fn test_place_for_current_returns_root_cwd_when_no_process_context() {
        // In test environments there is no scheduler hook registered, so
        // `process_info_current()` returns None and `cwd` defaults to "/".
        let place = place_for_current();
        assert_eq!(place.cwd, "/");
    }

    #[test]
    fn test_place_for_current_namespace_is_global() {
        let place = place_for_current();
        assert_eq!(place.namespace, "global");
    }

    #[test]
    fn test_place_for_current_root_is_slash() {
        let place = place_for_current();
        assert_eq!(place.root, "/");
    }

    #[test]
    fn test_place_for_current_as_text_is_valid() {
        let place = place_for_current();
        let text = place.as_text();
        assert!(text.contains("cwd: "), "unexpected: {text}");
        assert!(text.contains("namespace: global"), "unexpected: {text}");
        assert!(text.contains("root: /"), "unexpected: {text}");
    }
}
