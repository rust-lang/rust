//! Canonical public types for the `thingos.authority` schema kind.
//!
//! # Schema (v1 — Phase 7)
//!
//! ```text
//! /// A capability-based authority
//! kind thingos.authority = struct {
//!   name: string,
//!   capabilities: list<string>,
//! }
//! ```
//!
//! # What Authority is
//!
//! `Authority` is the canonical permission context in ThingOS.  It is the
//! explicit answer to the question **"under what power does this action occur?"**
//!
//! In Phase 7 its shape is intentionally minimal.  The `name` identifies the
//! acting authority (e.g. the executable name or a named service account).
//! The `capabilities` list carries any active capability strings; for most
//! processes in v1 this is empty.
//!
//! # What belongs under Authority
//!
//! * Who may open a resource
//! * Who may signal a task
//! * Who may map memory
//! * Who owns an operation
//! * What rights or roles are currently active
//!
//! Future phases will expand this to include uid/gid-like identity fields,
//! capability masks, delegation chains, and service-account or principal
//! bindings as those concepts are disentangled from `Process`.
//!
//! # Transitional mapping
//!
//! The current kernel `Process` structure is the *provisional* internal
//! backing for an `Authority`.  The `kernel::authority::bridge` module
//! translates `Process`-shaped credential and permission context into this
//! public type.
//!
//! | Canonical field          | Current kernel source                               |
//! |--------------------------|-----------------------------------------------------|
//! | `Authority::name`        | `ProcessSnapshot::name` (process/thread name)       |
//! | `Authority::capabilities`| *(empty — no capability field in `Process` yet)*    |
//!
//! # Note on `Process` as transitional backing
//!
//! The `Process` struct currently carries no explicit uid/gid, capability
//! mask, or service-account field.  Those fields will be added to `Process`
//! (or its Authority-shaped substructure introduced in Phase 5) and surfaced
//! here as the extraction matures.  The bridge documents all Process-carried
//! credential state that remains provisional pending fuller extraction.
//!
//! # Future direction
//!
//! * **Phase 8**: attach cwd / namespace bindings to a `Place`-shaped context.
//! * **Phase 9+**: uid/gid-like fields, capability masks, delegation chains.

extern crate alloc;

/// Canonical permission context for a running unit in ThingOS.
///
/// Corresponds to the `thingos.authority` schema kind (v1).  The kernel's
/// internal `Process` structure is the current transitional backing; the
/// `bridge` module in `kernel::authority` converts `Process`-shaped credential
/// state into this type.
///
/// This struct is intentionally minimal for v1.  Additional fields (uid/gid,
/// capability masks, delegation chains) will be added as the bridge matures
/// and the internal `Process` credential state is progressively extracted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Authority {
    /// The canonical name of this authority context.
    ///
    /// In Phase 7 this is derived from the process/thread name.  Future phases
    /// will refine this to a stable service-account or principal identifier.
    pub name: alloc::string::String,
    /// Active capability strings for this authority context.
    ///
    /// Empty in Phase 7 because `Process` carries no explicit capability mask.
    /// Future phases will populate this from the Process authority substructure
    /// introduced in Phase 5.
    pub capabilities: alloc::vec::Vec<alloc::string::String>,
}

impl Authority {
    /// Format as a human-readable text blob suitable for procfs.
    ///
    /// Output:
    /// ```text
    /// name: bristle
    /// capabilities: []
    /// ```
    /// or (once capabilities are populated):
    /// ```text
    /// name: my-service
    /// capabilities: [net_bind, read_fs]
    /// ```
    pub fn as_text(&self) -> alloc::string::String {
        let caps = if self.capabilities.is_empty() {
            alloc::string::String::from("[]")
        } else {
            alloc::format!("[{}]", self.capabilities.join(", "))
        };
        alloc::format!("name: {}\ncapabilities: {}\n", self.name, caps)
    }
}
