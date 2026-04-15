//! Canonical public types for the `thingos.group` schema kind.
//!
//! # Schema (v1 — Phase 4)
//!
//! ```text
//! kind thingos.group.kind = enum {
//!   Foreground,
//!   Coordination,
//! }
//!
//! kind thingos.group = struct {
//!   kind: thingos.group.kind,
//! }
//! ```
//!
//! # What Group is
//!
//! `Group` is the canonical coordination domain in ThingOS.  It represents the
//! role that a set of jobs plays with respect to control flow and coordination.
//!
//! In Phase 4 the shape is intentionally minimal.  Membership (which `Job`s
//! belong to this `Group`) is not yet part of the public type; it remains
//! internal while the coordination role itself becomes canonical.  Future
//! phases will progressively migrate:
//!
//! * signal fanout targets (`SIGCONT`, `SIGSTOP`, `SIGHUP`, …)
//! * TTY foreground ownership
//! * process-group and session semantics
//!
//! into `Group` terms, at which point membership and broadcast semantics will
//! join the public schema.
//!
//! # Why `Group` ≠ Unix process group
//!
//! Unix `pgid` is a narrow coordination hint that conflates:
//!
//! * terminal foreground ownership
//! * signal broadcast domains
//! * `waitpid(-1, …)` scoping
//!
//! `thingos.group` replaces all three with a single explicit coordination
//! object whose *role* (`GroupKind`) is declared rather than inferred from
//! POSIX rules.
//!
//! # Transitional mapping
//!
//! For now `kernel::group::bridge` maps the current Unix-shaped coordination
//! state into this canonical type:
//!
//! | Kernel source                                       | Canonical `Group`               |
//! |-----------------------------------------------------|---------------------------------|
//! | `Process::session_leader == true`                   | `GroupKind::Foreground`         |
//! | `ConsoleTtyState::foreground_pgid == Some(this.pgid)` | `GroupKind::Foreground`       |
//! | otherwise                                           | `GroupKind::Coordination`       |
//!
//! This mapping is provisional.  Do not assume it is semantically complete.
//! As signal routing and TTY ownership migrate into `Group` terms the bridge
//! will be updated and the internal Unix-shaped structures will be made
//! progressively obsolete.

/// The KindId generated for `thingos.group` by `kindc`.
///
/// Identifies the canonical coordination-domain schema kind.  Consumers that
/// need to distinguish a `Group`-shaped message payload can compare against
/// this constant.
pub const KIND_ID_THINGOS_GROUP: [u8; 16] = [
    0x2e, 0xc7, 0xf2, 0xca, 0x9d, 0x59, 0xe8, 0x5e,
    0x7f, 0x92, 0x7d, 0xd5, 0xd1, 0x5e, 0x01, 0x98,
];

/// The coordination role that a group plays in the system.
///
/// Corresponds to the `thingos.group.kind` schema kind (v1).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GroupKind {
    /// This group currently holds foreground control over a terminal.
    ///
    /// The canonical replacement for the Unix "foreground process group" concept.
    /// When a job's process group matches the TTY's foreground pgid, it belongs
    /// to a `Foreground` coordination domain.
    Foreground,
    /// A background or non-foreground coordination domain.
    ///
    /// Covers all other coordination groupings: background process groups,
    /// session-level broadcast targets, and future ThingOS-native coordination
    /// structures not tied to TTY ownership.
    Coordination,
}

impl GroupKind {
    /// Return a short human-readable label.
    pub fn as_str(self) -> &'static str {
        match self {
            GroupKind::Foreground => "Foreground",
            GroupKind::Coordination => "Coordination",
        }
    }
}

/// Canonical representation of a coordination domain.
///
/// Corresponds to the `thingos.group` schema kind (v1).  Membership
/// (the set of `Job`s that belong to this group) is not yet part of the
/// public schema; it remains internal in Phase 4.
///
/// The kernel's internal Unix-shaped process-group and session machinery backs
/// this type transitionally through `kernel::group::bridge`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Group {
    /// The coordination role of this group.
    pub kind: GroupKind,
}

impl Group {
    /// Format as a human-readable text blob suitable for procfs.
    ///
    /// Output:
    /// ```text
    /// kind: Foreground
    /// ```
    pub fn as_text(&self) -> alloc::string::String {
        alloc::format!("kind: {}\n", self.kind.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── GroupKind ────────────────────────────────────────────────────────────

    #[test]
    fn test_group_kind_as_str_foreground() {
        assert_eq!(GroupKind::Foreground.as_str(), "Foreground");
    }

    #[test]
    fn test_group_kind_as_str_coordination() {
        assert_eq!(GroupKind::Coordination.as_str(), "Coordination");
    }

    #[test]
    fn test_group_kind_equality() {
        assert_eq!(GroupKind::Foreground, GroupKind::Foreground);
        assert_eq!(GroupKind::Coordination, GroupKind::Coordination);
        assert_ne!(GroupKind::Foreground, GroupKind::Coordination);
    }

    // ── Group ────────────────────────────────────────────────────────────────

    #[test]
    fn test_group_carries_kind() {
        let g = Group { kind: GroupKind::Foreground };
        assert_eq!(g.kind, GroupKind::Foreground);
    }

    #[test]
    fn test_group_as_text_foreground() {
        let g = Group { kind: GroupKind::Foreground };
        let text = g.as_text();
        assert!(text.contains("kind: Foreground"), "unexpected: {}", text);
    }

    #[test]
    fn test_group_as_text_coordination() {
        let g = Group { kind: GroupKind::Coordination };
        let text = g.as_text();
        assert!(text.contains("kind: Coordination"), "unexpected: {}", text);
    }

    #[test]
    fn test_group_as_text_ends_with_newline() {
        let g = Group { kind: GroupKind::Foreground };
        assert!(g.as_text().ends_with('\n'));
    }

    #[test]
    fn test_group_equality() {
        assert_eq!(Group { kind: GroupKind::Foreground }, Group { kind: GroupKind::Foreground });
        assert_ne!(
            Group { kind: GroupKind::Foreground },
            Group { kind: GroupKind::Coordination }
        );
    }
}
