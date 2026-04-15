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

/// The KindId generated for `thingos.authority` by `kindc`.
///
/// Identifies the canonical permission-context schema kind.  Consumers that
/// need to distinguish an `Authority`-shaped message payload can compare
/// against this constant.
pub const KIND_ID_THINGOS_AUTHORITY: [u8; 16] = [
    0xc7, 0x7d, 0x6d, 0x64, 0xb9, 0xfd, 0x54, 0x04,
    0x23, 0x53, 0xfc, 0x38, 0x1c, 0x8c, 0xee, 0xbc,
];

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

#[cfg(test)]
mod tests {
    use super::*;

    // ── KIND_ID_THINGOS_AUTHORITY ─────────────────────────────────────────────

    #[test]
    fn test_kind_id_thingos_authority_constant() {
        // Verify the constant matches the kindc-generated value.
        let expected: [u8; 16] = [
            0xc7, 0x7d, 0x6d, 0x64, 0xb9, 0xfd, 0x54, 0x04,
            0x23, 0x53, 0xfc, 0x38, 0x1c, 0x8c, 0xee, 0xbc,
        ];
        assert_eq!(KIND_ID_THINGOS_AUTHORITY, expected);
    }

    // ── Authority construction ────────────────────────────────────────────────

    #[test]
    fn test_authority_carries_name() {
        let a = Authority {
            name: alloc::string::String::from("bristle"),
            capabilities: alloc::vec::Vec::new(),
        };
        assert_eq!(a.name, "bristle");
    }

    #[test]
    fn test_authority_carries_capabilities() {
        let a = Authority {
            name: alloc::string::String::from("svc"),
            capabilities: alloc::vec![
                alloc::string::String::from("net_bind"),
                alloc::string::String::from("read_fs"),
            ],
        };
        assert_eq!(a.capabilities.len(), 2);
        assert_eq!(a.capabilities[0], "net_bind");
        assert_eq!(a.capabilities[1], "read_fs");
    }

    // ── as_text ───────────────────────────────────────────────────────────────

    #[test]
    fn test_as_text_empty_capabilities() {
        let a = Authority {
            name: alloc::string::String::from("bristle"),
            capabilities: alloc::vec::Vec::new(),
        };
        let text = a.as_text();
        assert!(text.contains("name: bristle"), "unexpected: {text}");
        assert!(text.contains("capabilities: []"), "unexpected: {text}");
    }

    #[test]
    fn test_as_text_with_capabilities() {
        let a = Authority {
            name: alloc::string::String::from("my-service"),
            capabilities: alloc::vec![
                alloc::string::String::from("net_bind"),
                alloc::string::String::from("read_fs"),
            ],
        };
        let text = a.as_text();
        assert!(text.contains("name: my-service"), "unexpected: {text}");
        assert!(
            text.contains("capabilities: [net_bind, read_fs]"),
            "unexpected: {text}"
        );
    }

    #[test]
    fn test_as_text_ends_with_newline() {
        let a = Authority {
            name: alloc::string::String::from("x"),
            capabilities: alloc::vec::Vec::new(),
        };
        assert!(a.as_text().ends_with('\n'));
    }

    // ── equality ──────────────────────────────────────────────────────────────

    #[test]
    fn test_authority_equality_same() {
        let a = Authority {
            name: alloc::string::String::from("svc"),
            capabilities: alloc::vec::Vec::new(),
        };
        let b = Authority {
            name: alloc::string::String::from("svc"),
            capabilities: alloc::vec::Vec::new(),
        };
        assert_eq!(a, b);
    }

    #[test]
    fn test_authority_inequality_different_name() {
        let a = Authority {
            name: alloc::string::String::from("alpha"),
            capabilities: alloc::vec::Vec::new(),
        };
        let b = Authority {
            name: alloc::string::String::from("beta"),
            capabilities: alloc::vec::Vec::new(),
        };
        assert_ne!(a, b);
    }

    #[test]
    fn test_authority_inequality_different_capabilities() {
        let a = Authority {
            name: alloc::string::String::from("svc"),
            capabilities: alloc::vec![alloc::string::String::from("net_bind")],
        };
        let b = Authority {
            name: alloc::string::String::from("svc"),
            capabilities: alloc::vec::Vec::new(),
        };
        assert_ne!(a, b);
    }
}
