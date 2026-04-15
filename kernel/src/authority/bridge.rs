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
//!
//! # Guardrail: new authorization code entry points
//!
//! **New authorization checks must use the helpers in this module**, not raw
//! `Process` fields.  The canonical call sequence for a privileged syscall is:
//!
//! ```text
//! let authority = authority_for_current();
//! check_privilege(&authority, "reboot")?;
//! ```
//!
//! This keeps all authorization decision points visible and in one place,
//! making future migration to a fully-extracted `Authority` struct trivial.

use abi::errors::{Errno, SysResult};
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

/// Return the canonical [`Authority`] for the **currently running task**.
///
/// This is the **preferred entry point** for any new authorization check that
/// needs to know "under what power is this action occurring?".  Callers should
/// use this function instead of reaching into `Process` fields directly.
///
/// # Transitional behaviour
///
/// In Phase 7 this derives the `Authority` from the current thread name (via
/// the scheduler's name hook) and falls back to the process `exec_path` when
/// the thread name is not set.  When no process context is available (kernel
/// threads), the returned `Authority` uses the string `"kernel"` as its name
/// with no capabilities, representing unrestricted kernel-mode execution.
///
/// # Migration note
///
/// Once uid/gid-like fields or a capability mask are added to `Process`, this
/// function remains the **single entry point** — callers will transparently
/// receive a richer `Authority` without code changes at call sites.
pub fn authority_for_current() -> Authority {
    // PROVISIONAL: Derive authority name from the current thread name.
    // The scheduler's `current_task_name_current` hook returns a NUL-padded
    // [u8; 32] name stored on the Thread struct (not on Process).
    // Future phases will replace this with a stable principal identifier
    // sourced from an Authority-shaped substructure inside Process.
    let raw_name = unsafe { crate::sched::current_task_name_current() };
    let end = raw_name.iter().position(|&b| b == 0).unwrap_or(32);
    let name_str = core::str::from_utf8(&raw_name[..end]).unwrap_or("").trim();

    let name = if name_str.is_empty() {
        // Fall back to exec_path from ProcessInfo when thread name is not set.
        crate::sched::process_info_current()
            .map(|p| p.lock().exec_path.clone())
            .unwrap_or_default()
    } else {
        alloc::string::String::from(name_str)
    };

    Authority {
        name: if name.is_empty() {
            // Kernel threads have no ProcessInfo and no name — label them
            // explicitly so callers can identify privileged kernel context.
            alloc::string::String::from("kernel")
        } else {
            name
        },
        // PROVISIONAL: capabilities always empty; no capability field in Process yet.
        capabilities: alloc::vec::Vec::new(),
    }
}

/// Check whether an [`Authority`] holds a named privilege.
///
/// This is the **canonical gate** for privileged operations.  New
/// authorization checks must call this function instead of reading `Process`
/// fields directly.
///
/// # Transitional behaviour
///
/// In Phase 7 there is **no** explicit privilege model: `Process` carries no
/// uid/gid, role, or capability mask.  This function currently always returns
/// `Ok(())` so that the call sites compile and the pattern is established,
/// while clearly documenting that real enforcement is deferred.
///
/// The return type is `SysResult<()>` so that future phases can return
/// `Err(Errno::EPERM)` once a real privilege model is introduced, without
/// touching every call site.
///
/// # Usage
///
/// ```text
/// let authority = authority_for_current();
/// check_privilege(&authority, "reboot")?;
/// // ... proceed with privileged operation
/// ```
///
/// # Future direction
///
/// When a capability mask or role model is added to `Process` (or its
/// Authority-shaped substructure), this function will be updated to enforce
/// the check.  All call sites will automatically gain real enforcement without
/// code changes.
///
/// # PROVISIONAL
///
/// This function is a **transitional stub**.  Real privilege enforcement is not
/// yet implemented.  Do not rely on it for security decisions in production
/// until the `TODO(authority-enforcement)` marker below is resolved.
pub fn check_privilege(_authority: &Authority, _privilege: &str) -> SysResult<()> {
    // TODO(authority-enforcement): enforce capability/role check once
    // Process carries an explicit capability mask or role field.  For now
    // this is a no-op stub that establishes the call-site pattern.
    //
    // PROVISIONAL: all calls succeed in Phase 7.
    Ok(())
}
