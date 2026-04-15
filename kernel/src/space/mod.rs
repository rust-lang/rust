//! Space module: canonical address-space / virtual-memory domain object.
//!
//! # What this module provides
//!
//! This module introduces `Space` as a **first-class kernel object** — the
//! explicit owner of virtual memory state.  It is Phase 1 of the Space
//! extraction roadmap (see `docs/architecture/space-object.md`).
//!
//! ## Phase 1 goals (this module)
//!
//! * Introduce the `Space` kernel type that wraps the existing
//!   `ProcessAddressSpace` fields.
//! * Assign a stable `SpaceId` to every address space at creation time.
//! * Provide `kernel::space::bridge` as the **single conversion point** from
//!   the internal `ProcessAddressSpace` to the canonical `thingos::space::Space`
//!   public type.
//! * Make Space visible in kernel diagnostics (debug format, procfs).
//!
//! ## Transitional role of `ProcessAddressSpace`
//!
//! `ProcessAddressSpace` continues to live inside `Process.space` as the
//! **extraction seam**.  In this phase, `Space` wraps those same fields rather
//! than replacing them, so that:
//! - no Process-facing call sites need to change yet, and
//! - the canonical `Space` shape is established at system boundaries.
//!
//! Future phases will:
//! - move `ProcessAddressSpace` fields directly into `Space`
//! - replace `Process.space: ProcessAddressSpace` with
//!   `Process.space: Arc<Space>`
//! - allow multiple Tasks to share one `Space`
//!
//! ## What `Space` is
//!
//! See `docs/architecture/space-object.md` for the full design document.
//!
//! | Field / concern           | Owner in Phase 1                            |
//! |---------------------------|---------------------------------------------|
//! | `mappings: Arc<Mutex<MappingList>>` | `Space` (via `ProcessAddressSpace`) |
//! | `aspace_raw: u64`         | `Space` (via `ProcessAddressSpace`)         |
//! | `id: SpaceId`             | `Space` (new in this phase)                 |
//!
//! ## What `Space` is NOT
//!
//! - It is not a scheduler entity (that is `Thread<R>` / `Task`)
//! - It does not own lifecycle accounting (that is `Job`)
//! - It does not own credentials (that is `Authority`)
//! - It does not own cwd/namespace (that is `Place`)
//! - It does not own session/group (that is `Group`)

pub mod bridge;

use alloc::sync::Arc;
use core::sync::atomic::{AtomicU64, Ordering};
use spin::Mutex;
use thingos::space::SpaceId;

/// Monotonic counter for assigning stable `SpaceId`s.
///
/// Starts at 1 so that `SpaceId(0)` remains the sentinel "no space" value.
static NEXT_SPACE_ID: AtomicU64 = AtomicU64::new(1);

/// Allocate the next unique `SpaceId`.
///
/// IDs are assigned in monotonically increasing order starting from 1.
/// `SpaceId(0)` is reserved as `SpaceId::NONE`.
pub fn alloc_space_id() -> SpaceId {
    SpaceId(NEXT_SPACE_ID.fetch_add(1, Ordering::Relaxed))
}

/// First-class kernel address-space object.
///
/// A `Space` is the canonical owner of virtual-memory state for one or more
/// running tasks.  In Phase 1 it wraps the same fields that used to live in
/// `ProcessAddressSpace` and adds a stable identity via [`SpaceId`].
///
/// # Ownership model (Phase 1 — transitional)
///
/// In Phase 1, a `Space` is created alongside each `Process` and stored as
/// `Process.space_obj: Arc<Space>`.  The `mappings` arc inside `Space` is
/// the **same** `Arc<Mutex<MappingList>>` that lives in
/// `Process.space.mappings`, so all existing code that reaches mappings
/// through `Process.space.mappings` continues to work without changes.
///
/// In future phases:
/// - `Process.space: ProcessAddressSpace` is replaced by `Arc<Space>`.
/// - Multiple threads can hold `Arc<Space>` to share one address space.
/// - exec is modelled as Space replacement rather than in-place mutation.
///
/// # Locking rules
///
/// - `Space.mappings` is protected by its own `Mutex<MappingList>`; callers
///   must not hold the `Process` mutex while acquiring `mappings`.
/// - The `Space` itself is accessed via `Arc<Space>`; no additional lock is
///   needed to read `id` or `aspace_raw`.
/// - `aspace_raw` is written only during exec/spawn (single writer, inside
///   the `Process` mutex); it is read by the scheduler's fast path.
pub struct Space {
    /// Stable diagnostic identifier for this space.
    ///
    /// Assigned once at creation time via [`alloc_space_id`].  Unique within
    /// a boot session.  `SpaceId(0)` is never assigned to a real `Space`.
    pub id: SpaceId,

    /// VM mapping list — the authoritative list of virtual regions.
    ///
    /// Shared via `Arc` with every `Thread` that runs in this `Space` so the
    /// scheduler's per-CPU `CURRENT_MAPPINGS` cache can be updated without
    /// locking `Process` on every context switch.
    pub mappings: Arc<Mutex<crate::memory::mappings::MappingList>>,

    /// Architecture-specific page-table token.
    ///
    /// Stores the page-table root in an architecture-neutral `u64` form (see
    /// [`crate::BootTasking::aspace_to_raw`]).  Written during spawn/exec;
    /// read by the scheduler on context switch.
    pub aspace_raw: u64,
}

impl Space {
    /// Create a new, empty `Space` with a freshly allocated [`SpaceId`].
    ///
    /// The returned `Space` has no mapped regions and a zero page-table token.
    /// This is the state before ELF loading assigns a real address space.
    pub fn new_empty() -> Arc<Self> {
        Arc::new(Space {
            id: alloc_space_id(),
            mappings: Arc::new(Mutex::new(
                crate::memory::mappings::MappingList::new(),
            )),
            aspace_raw: 0,
        })
    }

    /// Create a `Space` from existing `ProcessAddressSpace` fields plus a new
    /// `SpaceId`.
    ///
    /// This is the **primary construction path** in Phase 1: called when a
    /// `ProcessAddressSpace` already exists (e.g. after ELF loading) and we
    /// want to wrap it in a first-class `Space` object.
    ///
    /// The `mappings` `Arc` is **shared** — the same underlying
    /// `MappingList` is owned by both the caller's `ProcessAddressSpace` and
    /// this `Space`, so mutations through either path are immediately visible
    /// through both.
    pub fn from_process_address_space(
        pas: &crate::task::ProcessAddressSpace,
    ) -> Arc<Self> {
        Arc::new(Space {
            id: alloc_space_id(),
            mappings: Arc::clone(&pas.mappings),
            aspace_raw: pas.aspace_raw,
        })
    }

    /// Return the number of mapped virtual regions at this moment.
    ///
    /// Acquires `mappings` briefly.  Suitable for diagnostics; not a stable
    /// value for decisions.
    pub fn mapping_count(&self) -> usize {
        self.mappings.lock().regions.len()
    }
}

impl core::fmt::Debug for Space {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Space")
            .field("id", &self.id)
            .field("mapping_count", &self.mapping_count())
            .field("aspace_raw", &self.aspace_raw)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::mappings::MappingList;

    // ── alloc_space_id ────────────────────────────────────────────────────────

    #[test]
    fn test_alloc_space_id_nonzero() {
        let id = alloc_space_id();
        assert!(!id.is_none(), "alloc_space_id must not return SpaceId::NONE");
    }

    #[test]
    fn test_alloc_space_id_monotone() {
        let a = alloc_space_id();
        let b = alloc_space_id();
        assert!(a < b, "space IDs must be strictly increasing");
    }

    // ── Space::new_empty ──────────────────────────────────────────────────────

    #[test]
    fn test_new_empty_has_nonzero_id() {
        let s = Space::new_empty();
        assert!(!s.id.is_none());
    }

    #[test]
    fn test_new_empty_has_zero_mappings() {
        let s = Space::new_empty();
        assert_eq!(s.mapping_count(), 0);
    }

    #[test]
    fn test_new_empty_has_zero_aspace_raw() {
        let s = Space::new_empty();
        assert_eq!(s.aspace_raw, 0);
    }

    // ── Space::from_process_address_space ─────────────────────────────────────

    #[test]
    fn test_from_process_address_space_shares_mappings_arc() {
        let mappings_arc = Arc::new(Mutex::new(MappingList::new()));
        let pas = crate::task::ProcessAddressSpace::from_parts(
            Arc::clone(&mappings_arc),
            0xDEAD_BEEF,
        );
        let space = Space::from_process_address_space(&pas);

        // The two Arcs should point to the same allocation.
        assert!(
            Arc::ptr_eq(&space.mappings, &mappings_arc),
            "Space must share the same Arc<Mutex<MappingList>> as ProcessAddressSpace"
        );
    }

    #[test]
    fn test_from_process_address_space_copies_aspace_raw() {
        let pas = crate::task::ProcessAddressSpace::from_parts(
            Arc::new(Mutex::new(MappingList::new())),
            0xCAFE_0000,
        );
        let space = Space::from_process_address_space(&pas);
        assert_eq!(space.aspace_raw, 0xCAFE_0000);
    }

    #[test]
    fn test_from_process_address_space_has_nonzero_id() {
        let pas = crate::task::ProcessAddressSpace::empty();
        let space = Space::from_process_address_space(&pas);
        assert!(!space.id.is_none());
    }

    // ── mapping_count reflects mutations ──────────────────────────────────────

    #[test]
    fn test_mapping_count_reflects_insertions() {
        let space = Space::new_empty();
        assert_eq!(space.mapping_count(), 0);

        space.mappings.lock().insert(abi::vm::VmRegionInfo {
            start: 0x1000,
            end: 0x2000,
            prot: abi::vm::VmProt::READ,
            ..Default::default()
        });
        assert_eq!(space.mapping_count(), 1);
    }

    // ── Arc sharing: mutation via ProcessAddressSpace is visible via Space ────

    #[test]
    fn test_shared_arc_mutations_visible_from_space() {
        let mappings_arc = Arc::new(Mutex::new(MappingList::new()));
        let pas = crate::task::ProcessAddressSpace::from_parts(
            Arc::clone(&mappings_arc),
            0,
        );
        let space = Space::from_process_address_space(&pas);

        // Mutate through the ProcessAddressSpace mappings arc.
        pas.mappings.lock().insert(abi::vm::VmRegionInfo {
            start: 0x4000,
            end: 0x5000,
            prot: abi::vm::VmProt::READ | abi::vm::VmProt::WRITE,
            ..Default::default()
        });

        // Mutation must be visible through Space.
        assert_eq!(
            space.mapping_count(),
            1,
            "Space must see mutations made through the shared mappings Arc"
        );
    }
}
