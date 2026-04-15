//! Bridge layer: kernel `Space` / `ProcessAddressSpace` → canonical
//! `thingos::space::Space`.
//!
//! # Purpose
//!
//! This module is the **single conversion point** from the kernel's internal
//! address-space representation (`Space` / `ProcessAddressSpace`) to the
//! schema-generated canonical `thingos::space::Space` public type.  All
//! space-facing public paths (procfs, diagnostics, future ABI) should go
//! through here rather than reading internal fields directly.
//!
//! # Migration inventory — Process VM / address-space responsibilities
//!
//! The table below inventories every field in the current `Process` struct that
//! answers "where does memory live?" questions and maps it to its intended
//! future canonical home.
//!
//! | `Process` / `ProcessAddressSpace` field | VM role                            | Intended `Space` mapping   | Status       |
//! |-----------------------------------------|------------------------------------|----------------------------|--------------|
//! | `space.mappings`                        | VM region list                     | `Space::mappings`          | **Bridged**  |
//! | `space.aspace_raw`                      | Architecture page-table token      | `Space::aspace_raw`        | **Bridged**  |
//! | `space_obj`                             | First-class `Arc<Space>` wrapper   | `Space::id`                | **New**      |
//! | *(no COW lineage yet)*                  | Clone lineage metadata             | Future `Space` field       | Not yet added|
//! | *(no fault accounting yet)*             | Page-fault counters                | Future `Space` field       | Not yet added|
//!
//! # Transitional mapping
//!
//! | Internal source                              | Canonical `thingos::space::Space` field | Notes                                        |
//! |----------------------------------------------|-----------------------------------------|----------------------------------------------|
//! | `Space::id`                                  | `id`                                    | Stable `SpaceId` assigned at creation        |
//! | `Space::mapping_count()`                     | `mapping_count`                         | Snapshot at bridge-call time                 |
//! | `Arc::strong_count(&Space::mappings) − 1`    | `sharing_count`                         | Approx; 0 is possible transiently            |
//!
//! # What is not yet replaced
//!
//! * COW lineage / clone ancestry — no such field in `ProcessAddressSpace` yet
//! * Page-fault accounting counters — not yet present
//! * Explicit exec-replacement semantics — exec still mutates in place
//!
//! # Future direction
//!
//! When `Process.space: ProcessAddressSpace` is replaced by
//! `Process.space_obj: Arc<Space>`, this bridge will derive the canonical
//! `Space` directly from the `Arc<Space>` without going through
//! `ProcessAddressSpace`.  Call sites will not change.
//!
//! # Entry points
//!
//! | Code context                              | Preferred function                  |
//! |-------------------------------------------|-------------------------------------|
//! | Has an `Arc<crate::space::Space>`         | `space_from_arc`                    |
//! | Has a `ProcessAddressSpace` snapshot      | `space_from_process_address_space`  |
//! | Has a `ProcessSnapshot`                   | `space_from_snapshot`               |
//! | Needs the current task's space            | `space_for_current`                 |

use alloc::sync::Arc;
use thingos::space::{Space as PublicSpace, SpaceId};

// ── space_from_arc ────────────────────────────────────────────────────────────

/// Build a canonical [`thingos::space::Space`] from a kernel [`crate::space::Space`] arc.
///
/// This is the **preferred** entry point when you already have an
/// `Arc<crate::space::Space>` — it derives `sharing_count` from the live
/// reference count and calls `mapping_count()` to snapshot the region count.
///
/// # Transitional note
///
/// In Phase 1 the `Arc<Space>` lives in `Process.space.space_obj`.  Once
/// `ProcessAddressSpace` is fully replaced by `Arc<Space>`, this function
/// becomes the only bridge entry point needed.
pub fn space_from_arc(space: &Arc<crate::space::Space>) -> PublicSpace {
    // sharing_count: number of OTHER holders = (strong_count − 1).
    // We subtract 1 because the caller's own clone is included in strong_count.
    // Using `saturating_sub` to guard against the transient zero-refcount
    // window that can occur during teardown.
    let sharing_count = (Arc::strong_count(space) as u32).saturating_sub(1);
    let mapping_count = space.mapping_count() as u32;

    PublicSpace {
        id: space.id,
        mapping_count,
        sharing_count,
    }
}

// ── space_from_process_address_space ─────────────────────────────────────────

/// Build a canonical [`thingos::space::Space`] from a
/// [`crate::task::ProcessAddressSpace`] snapshot.
///
/// This is the **transitional** path used in Phase 1 when a caller holds a
/// `ProcessAddressSpace` but wants a snapshot using the `space_obj.id`
/// embedded within it.  The `space_id` parameter may be supplied explicitly
/// to override the embedded id (e.g. for testing), or you can use
/// `pas.space_obj.id` as the natural choice.
///
/// # Note on sharing_count
///
/// When called with just a `ProcessAddressSpace`, the sharing count is
/// derived from `Arc::strong_count(&pas.mappings)`.  This includes the
/// copy held by each `Thread.mappings`; subtract 1 for the `Process.space`
/// copy itself.
pub fn space_from_process_address_space(
    pas: &crate::task::ProcessAddressSpace,
    space_id: SpaceId,
) -> PublicSpace {
    let mapping_count = pas.mappings.lock().regions.len() as u32;
    // strong_count includes: Process.space, space_obj, every Thread.mappings.
    // We use saturating_sub(1) to represent "other holders besides this one".
    let sharing_count =
        (Arc::strong_count(&pas.mappings) as u32).saturating_sub(1);

    PublicSpace {
        id: space_id,
        mapping_count,
        sharing_count,
    }
}

// ── space_from_snapshot ───────────────────────────────────────────────────────

/// Build a canonical [`thingos::space::Space`] from a
/// [`crate::sched::hooks::ProcessSnapshot`].
///
/// This is the entry point for procfs / diagnostic code that already has a
/// `ProcessSnapshot` and needs to produce the canonical `Space` view.
///
/// In Phase 1 the snapshot carries `space_id`, `space_mapping_count`, and
/// `space_sharing_count` fields that are populated at snapshot time from
/// `Process.space_obj` and `Process.space`.
pub fn space_from_snapshot(
    snapshot: &crate::sched::hooks::ProcessSnapshot,
) -> PublicSpace {
    PublicSpace {
        id: snapshot.space_id,
        mapping_count: snapshot.space_mapping_count,
        sharing_count: snapshot.space_sharing_count,
    }
}

// ── space_for_current ─────────────────────────────────────────────────────────

/// Return the canonical [`thingos::space::Space`] for the **currently running task**.
///
/// Derives the space from the current task's owning `Process.space.space_obj`.
/// Returns a minimal placeholder `Space` for kernel-only threads that have
/// no `ProcessInfo`.
pub fn space_for_current() -> PublicSpace {
    if let Some(pinfo_arc) = crate::sched::process_info_current() {
        let pinfo = pinfo_arc.lock();
        space_from_arc(&pinfo.space.space_obj)
    } else {
        // Kernel thread — no user address space.
        PublicSpace {
            id: SpaceId::NONE,
            mapping_count: 0,
            sharing_count: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::mappings::MappingList;
    use crate::space::Space;
    use crate::task::ProcessAddressSpace;

    // ── space_from_arc ────────────────────────────────────────────────────────

    #[test]
    fn test_space_from_arc_id_propagated() {
        let space = Space::new_empty();
        let canonical = space_from_arc(&space);
        assert_eq!(canonical.id, space.id);
        assert!(!canonical.id.is_none());
    }

    #[test]
    fn test_space_from_arc_zero_mappings() {
        let space = Space::new_empty();
        let canonical = space_from_arc(&space);
        assert_eq!(canonical.mapping_count, 0);
    }

    #[test]
    fn test_space_from_arc_mapping_count_after_insert() {
        let space = Space::new_empty();
        space.mappings.lock().insert(abi::vm::VmRegionInfo {
            start: 0x1000,
            end: 0x2000,
            prot: abi::vm::VmProt::READ,
            ..Default::default()
        });
        let canonical = space_from_arc(&space);
        assert_eq!(canonical.mapping_count, 1);
    }

    #[test]
    fn test_space_from_arc_sharing_count_single_holder() {
        let space = Space::new_empty();
        // Only one Arc clone exists (the `space` binding above).
        // sharing_count = strong_count − 1 = 1 − 1 = 0.
        let canonical = space_from_arc(&space);
        assert_eq!(canonical.sharing_count, 0);
    }

    #[test]
    fn test_space_from_arc_sharing_count_two_holders() {
        let space = Space::new_empty();
        let _second = Arc::clone(&space);
        // strong_count = 2; sharing_count = 2 − 1 = 1.
        let canonical = space_from_arc(&space);
        assert_eq!(canonical.sharing_count, 1);
    }

    // ── space_from_process_address_space ──────────────────────────────────────

    #[test]
    fn test_space_from_pas_id_propagated() {
        let pas = ProcessAddressSpace::empty();
        let id = SpaceId(42);
        let canonical = space_from_process_address_space(&pas, id);
        assert_eq!(canonical.id, SpaceId(42));
    }

    #[test]
    fn test_space_from_pas_zero_mappings() {
        let pas = ProcessAddressSpace::empty();
        let canonical = space_from_process_address_space(&pas, SpaceId(1));
        assert_eq!(canonical.mapping_count, 0);
    }

    #[test]
    fn test_space_from_pas_mapping_count_reflects_list() {
        let mappings = Arc::new(spin::Mutex::new(MappingList::new()));
        {
            let mut ml = mappings.lock();
            ml.insert(abi::vm::VmRegionInfo {
                start: 0x3000,
                end: 0x4000,
                prot: abi::vm::VmProt::READ | abi::vm::VmProt::WRITE,
                ..Default::default()
            });
        }
        let pas = ProcessAddressSpace::from_parts(mappings, 0);
        let canonical = space_from_process_address_space(&pas, SpaceId(1));
        assert_eq!(canonical.mapping_count, 1);
    }

    // ── space_from_snapshot ───────────────────────────────────────────────────

    #[test]
    fn test_space_from_snapshot_fields_forwarded() {
        use crate::sched::hooks::ProcessSnapshot;
        use crate::task::TaskState;

        let snap = ProcessSnapshot {
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
            cwd: alloc::string::String::from("/"),
            namespace_label: alloc::string::String::from("global"),
            thread_states: alloc::vec![TaskState::Runnable],
            space_id: SpaceId(77),
            space_mapping_count: 5,
            space_sharing_count: 2,
        };

        let canonical = space_from_snapshot(&snap);
        assert_eq!(canonical.id, SpaceId(77));
        assert_eq!(canonical.mapping_count, 5);
        assert_eq!(canonical.sharing_count, 2);
    }

    #[test]
    fn test_space_from_snapshot_none_id_for_kernel_thread() {
        use crate::sched::hooks::ProcessSnapshot;
        use crate::task::TaskState;

        let snap = ProcessSnapshot {
            pid: 0,
            ppid: 0,
            tid: 0,
            name: alloc::string::String::from("ktask"),
            state: TaskState::Runnable,
            argv: alloc::vec::Vec::new(),
            exec_path: alloc::string::String::new(),
            exit_code: None,
            pgid: 0,
            sid: 0,
            session_leader: false,
            cwd: alloc::string::String::from("/"),
            namespace_label: alloc::string::String::from("global"),
            thread_states: alloc::vec::Vec::new(),
            space_id: SpaceId::NONE,
            space_mapping_count: 0,
            space_sharing_count: 0,
        };

        let canonical = space_from_snapshot(&snap);
        assert!(canonical.id.is_none());
        assert_eq!(canonical.mapping_count, 0);
        assert_eq!(canonical.sharing_count, 0);
    }
}
