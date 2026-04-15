//! Canonical public types for the `thingos.space` schema kind.
//!
//! # Conceptual mapping
//!
//! `Space` is the Janix replacement for the implicit address-space ownership
//! that currently lives inside the Unix `Process` concept.
//! See `docs/architecture/space-object.md` for the full design document and
//! `docs/migration/concept-mapping.md` §2 (Address Space → Space) for the
//! canonical legacy→Janix mapping.
//!
//! # Schema (v1)
//!
//! ```text
//! kind thingos.space.id = u64
//!
//! kind thingos.space = struct {
//!   id:            thingos.space.id,
//!   mapping_count: u32,
//!   sharing_count: u32,
//! }
//! ```
//!
//! # What Space is
//!
//! `Space` is the canonical *virtual memory identity* in ThingOS.  It is the
//! explicit answer to the question **"where does memory live, and under what
//! virtual mapping rules?"**
//!
//! A Space owns (or anchors):
//! - the page-table root / architecture VM context
//! - the set of mapped virtual regions (anonymous, file-backed, device)
//! - mapping permissions and attributes
//! - COW/clone lineage metadata (future)
//! - fault/accounting state (future)
//!
//! A Space does **not** imply:
//! - scheduler identity (that is `Task`)
//! - lifecycle/exit tracking (that is `Job`)
//! - credentials / capabilities (that is `Authority`)
//! - cwd / namespace (that is `Place`)
//! - session / process group (that is `Group`)
//!
//! # Transitional mapping
//!
//! The current kernel backing is `kernel::task::ProcessAddressSpace` inside
//! `Process.space`.  The bridge module `kernel::space::bridge` is the single
//! conversion point from that internal representation to this public type.
//!
//! | Canonical field   | Current kernel source                             |
//! |-------------------|---------------------------------------------------|
//! | `id`              | Stable `SpaceId` counter (new; assigned at spawn) |
//! | `mapping_count`   | `Process.space.mappings.lock().regions.len()`     |
//! | `sharing_count`   | `Arc::strong_count(&Process.space.mappings)` − 1  |
//!
//! # Future direction
//!
//! * Introduce `Arc<Space>` inside the kernel so multiple `Task`s can share
//!   one `Space` without going through `Process`.
//! * Expose `Space` via a file descriptor / handle for controlled sharing.
//! * Add exec-replacement semantics: `exec` atomically replaces the current
//!   `Space` rather than mutating fields in place.

extern crate alloc;

/// The KindId generated for `thingos.space` by `kindc`.
///
/// Identifies the canonical address-space schema kind.  Consumers that need
/// to distinguish a `Space`-shaped message payload can compare against this
/// constant.
pub const KIND_ID_THINGOS_SPACE: [u8; 16] = [
    0x3a, 0x7f, 0x9e, 0x12, 0xd4, 0x56, 0x4b, 0x8c,
    0xa1, 0x0f, 0x3c, 0x77, 0x25, 0xe9, 0xb3, 0x41,
];

/// Opaque identifier for a `Space` kernel object.
///
/// `SpaceId` is a monotonically-increasing counter assigned when a `Space`
/// is created.  It is unique within a single boot session and is suitable
/// for diagnostics, debug output, and future ABI plumbing.
///
/// `SpaceId(0)` is reserved as "no space" / "kernel space" and will never
/// be assigned to a user-created `Space`.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct SpaceId(pub u64);

impl SpaceId {
    /// The sentinel value representing "no space" (e.g. for kernel-only threads).
    pub const NONE: SpaceId = SpaceId(0);

    /// Return the raw numeric identifier.
    pub fn as_u64(self) -> u64 {
        self.0
    }

    /// Return `true` if this is the sentinel "no space" value.
    pub fn is_none(self) -> bool {
        self.0 == 0
    }
}

impl core::fmt::Display for SpaceId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "space:{}", self.0)
    }
}

/// Canonical public representation of an address-space / virtual-memory domain.
///
/// Corresponds to the `thingos.space` schema kind (v1).  The kernel's internal
/// `ProcessAddressSpace` structure is the current transitional backing; the
/// `kernel::space::bridge` module converts that state into this public type.
///
/// # Invariants
///
/// - `id` is non-zero for any real (user-visible) `Space`.
/// - `mapping_count` reflects the number of distinct mapped virtual regions at
///   snapshot time.  It may change concurrently; treat it as a best-effort
///   diagnostic value rather than an authoritative count.
/// - `sharing_count` reports how many `Task`s (or equivalent holders) currently
///   share this `Space`.  A count of `1` means the space is private to one
///   task; a count `> 1` means multiple tasks share the same address space.
///   A count of `0` is possible transiently during teardown.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Space {
    /// Unique diagnostic identifier for this space within a boot session.
    pub id: SpaceId,
    /// Number of distinct mapped virtual regions at snapshot time.
    pub mapping_count: u32,
    /// Number of tasks (or equivalent) currently sharing this space.
    ///
    /// `1` means private; `> 1` means shared.
    pub sharing_count: u32,
}

impl Space {
    /// Format as a human-readable text blob suitable for procfs / debug output.
    ///
    /// Output format:
    /// ```text
    /// id: space:42
    /// mapping_count: 7
    /// sharing_count: 1
    /// ```
    pub fn as_text(&self) -> alloc::string::String {
        alloc::format!(
            "id: {}\nmapping_count: {}\nsharing_count: {}\n",
            self.id,
            self.mapping_count,
            self.sharing_count,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_space(id: u64, mapping_count: u32, sharing_count: u32) -> Space {
        Space {
            id: SpaceId(id),
            mapping_count,
            sharing_count,
        }
    }

    // ── SpaceId ───────────────────────────────────────────────────────────────

    #[test]
    fn test_space_id_none_is_zero() {
        assert_eq!(SpaceId::NONE.as_u64(), 0);
    }

    #[test]
    fn test_space_id_is_none_true_for_zero() {
        assert!(SpaceId::NONE.is_none());
    }

    #[test]
    fn test_space_id_is_none_false_for_nonzero() {
        assert!(!SpaceId(1).is_none());
        assert!(!SpaceId(42).is_none());
    }

    #[test]
    fn test_space_id_display() {
        let id = SpaceId(7);
        let s = alloc::format!("{}", id);
        assert_eq!(s, "space:7");
    }

    #[test]
    fn test_space_id_ordering() {
        assert!(SpaceId(1) < SpaceId(2));
        assert!(SpaceId(10) > SpaceId(5));
        assert_eq!(SpaceId(3), SpaceId(3));
    }

    // ── Space::as_text ────────────────────────────────────────────────────────

    #[test]
    fn test_as_text_contains_id() {
        let s = make_space(42, 3, 1);
        let text = s.as_text();
        assert!(text.contains("id: space:42"), "unexpected: {text}");
    }

    #[test]
    fn test_as_text_contains_mapping_count() {
        let s = make_space(1, 7, 1);
        let text = s.as_text();
        assert!(text.contains("mapping_count: 7"), "unexpected: {text}");
    }

    #[test]
    fn test_as_text_contains_sharing_count() {
        let s = make_space(1, 7, 2);
        let text = s.as_text();
        assert!(text.contains("sharing_count: 2"), "unexpected: {text}");
    }

    #[test]
    fn test_as_text_ends_with_newline() {
        let s = make_space(1, 0, 1);
        assert!(s.as_text().ends_with('\n'));
    }

    #[test]
    fn test_as_text_full_format() {
        let s = make_space(5, 2, 1);
        assert_eq!(s.as_text(), "id: space:5\nmapping_count: 2\nsharing_count: 1\n");
    }

    // ── Space equality ────────────────────────────────────────────────────────

    #[test]
    fn test_space_equality_same() {
        let a = make_space(1, 3, 1);
        let b = make_space(1, 3, 1);
        assert_eq!(a, b);
    }

    #[test]
    fn test_space_inequality_different_id() {
        let a = make_space(1, 3, 1);
        let b = make_space(2, 3, 1);
        assert_ne!(a, b);
    }

    #[test]
    fn test_space_clone_is_equal() {
        let a = make_space(10, 4, 2);
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── KIND_ID constant sanity ───────────────────────────────────────────────

    #[test]
    fn test_kind_id_is_non_zero() {
        assert!(KIND_ID_THINGOS_SPACE.iter().any(|&b| b != 0));
    }
}
