use std::cell::RefCell;
use std::collections::hash_map::Entry;

use log::trace;
use rand::Rng;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_target::abi::{HasDataLayout, Size};

use crate::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ProvenanceMode {
    /// Int2ptr casts return pointers with "wildcard" provenance
    /// that basically matches that of all exposed pointers
    /// (and SB tags, if enabled).
    Permissive,
    /// Int2ptr casts return pointers with an invalid provenance,
    /// i.e., not valid for any memory access.
    Strict,
    /// Int2ptr casts determine the allocation they point to at cast time.
    /// All allocations are considered exposed.
    Legacy,
}

pub type GlobalState = RefCell<GlobalStateInner>;

#[derive(Clone, Debug)]
pub struct GlobalStateInner {
    /// This is used as a map between the address of each allocation and its `AllocId`.
    /// It is always sorted
    int_to_ptr_map: Vec<(u64, AllocId)>,
    /// The base address for each allocation.  We cannot put that into
    /// `AllocExtra` because function pointers also have a base address, and
    /// they do not have an `AllocExtra`.
    /// This is the inverse of `int_to_ptr_map`.
    base_addr: FxHashMap<AllocId, u64>,
    /// Whether an allocation has been exposed or not. This cannot be put
    /// into `AllocExtra` for the same reason as `base_addr`.
    exposed: FxHashSet<AllocId>,
    /// This is used as a memory address when a new pointer is casted to an integer. It
    /// is always larger than any address that was previously made part of a block.
    next_base_addr: u64,
    /// The provenance to use for int2ptr casts
    provenance_mode: ProvenanceMode,
}

impl GlobalStateInner {
    pub fn new(config: &MiriConfig) -> Self {
        GlobalStateInner {
            int_to_ptr_map: Vec::default(),
            base_addr: FxHashMap::default(),
            exposed: FxHashSet::default(),
            next_base_addr: STACK_ADDR,
            provenance_mode: config.provenance_mode,
        }
    }
}

impl<'mir, 'tcx> GlobalStateInner {
    // Returns the exposed `AllocId` that corresponds to the specified addr,
    // or `None` if the addr is out of bounds
    fn alloc_id_from_addr(ecx: &MiriEvalContext<'mir, 'tcx>, addr: u64) -> Option<AllocId> {
        let global_state = ecx.machine.intptrcast.borrow();
        assert!(global_state.provenance_mode != ProvenanceMode::Strict);

        let pos = global_state.int_to_ptr_map.binary_search_by_key(&addr, |(addr, _)| *addr);

        let alloc_id = match pos {
            Ok(pos) => Some(global_state.int_to_ptr_map[pos].1),
            Err(0) => None,
            Err(pos) => {
                // This is the largest of the adresses smaller than `int`,
                // i.e. the greatest lower bound (glb)
                let (glb, alloc_id) = global_state.int_to_ptr_map[pos - 1];
                // This never overflows because `addr >= glb`
                let offset = addr - glb;
                // If the offset exceeds the size of the allocation, don't use this `alloc_id`.

                if offset
                    <= ecx
                        .get_alloc_size_and_align(alloc_id, AllocCheck::MaybeDead)
                        .unwrap()
                        .0
                        .bytes()
                {
                    Some(alloc_id)
                } else {
                    None
                }
            }
        }?;

        // In legacy mode, we consider all allocations exposed.
        if global_state.provenance_mode == ProvenanceMode::Legacy
            || global_state.exposed.contains(&alloc_id)
        {
            Some(alloc_id)
        } else {
            None
        }
    }

    pub fn expose_ptr(ecx: &mut MiriEvalContext<'mir, 'tcx>, alloc_id: AllocId, sb: SbTag) {
        let global_state = ecx.machine.intptrcast.get_mut();
        // In legacy and strict mode, we don't need this, so we can save some cycles
        // by not tracking it.
        if global_state.provenance_mode == ProvenanceMode::Permissive {
            trace!("Exposing allocation id {alloc_id:?}");
            global_state.exposed.insert(alloc_id);
            if ecx.machine.stacked_borrows.is_some() {
                ecx.expose_tag(alloc_id, sb);
            }
        }
    }

    pub fn ptr_from_addr_transmute(
        ecx: &MiriEvalContext<'mir, 'tcx>,
        addr: u64,
    ) -> Pointer<Option<Tag>> {
        trace!("Transmuting 0x{:x} to a pointer", addr);

        if ecx.machine.allow_ptr_int_transmute {
            // When we allow transmutes, treat them like casts.
            Self::ptr_from_addr_cast(ecx, addr)
        } else {
            // We consider transmuted pointers to be "invalid" (`None` provenance).
            Pointer::new(None, Size::from_bytes(addr))
        }
    }

    pub fn ptr_from_addr_cast(
        ecx: &MiriEvalContext<'mir, 'tcx>,
        addr: u64,
    ) -> Pointer<Option<Tag>> {
        trace!("Casting 0x{:x} to a pointer", addr);

        let global_state = ecx.machine.intptrcast.borrow();

        match global_state.provenance_mode {
            ProvenanceMode::Legacy => {
                // Determine the allocation this points to at cast time.
                let alloc_id = Self::alloc_id_from_addr(ecx, addr);
                Pointer::new(
                    alloc_id.map(|alloc_id| {
                        Tag::Concrete(ConcreteTag { alloc_id, sb: SbTag::Untagged })
                    }),
                    Size::from_bytes(addr),
                )
            }
            ProvenanceMode::Strict => {
                // We don't support int2ptr casts in this mode (i.e., we treat them like
                // transmutes).
                Pointer::new(None, Size::from_bytes(addr))
            }
            ProvenanceMode::Permissive => {
                // This is how wildcard pointers are born.
                Pointer::new(Some(Tag::Wildcard), Size::from_bytes(addr))
            }
        }
    }

    fn alloc_base_addr(ecx: &MiriEvalContext<'mir, 'tcx>, alloc_id: AllocId) -> u64 {
        let mut global_state = ecx.machine.intptrcast.borrow_mut();
        let global_state = &mut *global_state;

        match global_state.base_addr.entry(alloc_id) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                // There is nothing wrong with a raw pointer being cast to an integer only after
                // it became dangling.  Hence `MaybeDead`.
                let (size, align) =
                    ecx.get_alloc_size_and_align(alloc_id, AllocCheck::MaybeDead).unwrap();

                // This allocation does not have a base address yet, pick one.
                // Leave some space to the previous allocation, to give it some chance to be less aligned.
                let slack = {
                    let mut rng = ecx.machine.rng.borrow_mut();
                    // This means that `(global_state.next_base_addr + slack) % 16` is uniformly distributed.
                    rng.gen_range(0..16)
                };
                // From next_base_addr + slack, round up to adjust for alignment.
                let base_addr = global_state.next_base_addr.checked_add(slack).unwrap();
                let base_addr = Self::align_addr(base_addr, align.bytes());
                entry.insert(base_addr);
                trace!(
                    "Assigning base address {:#x} to allocation {:?} (size: {}, align: {}, slack: {})",
                    base_addr,
                    alloc_id,
                    size.bytes(),
                    align.bytes(),
                    slack,
                );

                // Remember next base address.  Leave a gap of at least 1 to avoid two zero-sized allocations
                // having the same base address, and to avoid ambiguous provenance for the address between two
                // allocations (also see https://github.com/rust-lang/unsafe-code-guidelines/issues/313).
                let size_plus_1 = size.bytes().checked_add(1).unwrap();
                global_state.next_base_addr = base_addr.checked_add(size_plus_1).unwrap();
                // Given that `next_base_addr` increases in each allocation, pushing the
                // corresponding tuple keeps `int_to_ptr_map` sorted
                global_state.int_to_ptr_map.push((base_addr, alloc_id));

                base_addr
            }
        }
    }

    /// Convert a relative (tcx) pointer to an absolute address.
    pub fn rel_ptr_to_addr(ecx: &MiriEvalContext<'mir, 'tcx>, ptr: Pointer<AllocId>) -> u64 {
        let (alloc_id, offset) = ptr.into_parts(); // offset is relative (AllocId provenance)
        let base_addr = GlobalStateInner::alloc_base_addr(ecx, alloc_id);

        // Add offset with the right kind of pointer-overflowing arithmetic.
        let dl = ecx.data_layout();
        dl.overflowing_offset(base_addr, offset.bytes()).0
    }

    pub fn abs_ptr_to_rel(
        ecx: &MiriEvalContext<'mir, 'tcx>,
        ptr: Pointer<Tag>,
    ) -> Option<(AllocId, Size)> {
        let (tag, addr) = ptr.into_parts(); // addr is absolute (Tag provenance)

        let alloc_id = if let Tag::Concrete(concrete) = tag {
            concrete.alloc_id
        } else {
            // A wildcard pointer.
            assert_eq!(ecx.machine.intptrcast.borrow().provenance_mode, ProvenanceMode::Permissive);
            GlobalStateInner::alloc_id_from_addr(ecx, addr.bytes())?
        };

        let base_addr = GlobalStateInner::alloc_base_addr(ecx, alloc_id);

        // Wrapping "addr - base_addr"
        let dl = ecx.data_layout();
        let neg_base_addr = (base_addr as i64).wrapping_neg();
        Some((
            alloc_id,
            Size::from_bytes(dl.overflowing_signed_offset(addr.bytes(), neg_base_addr).0),
        ))
    }

    /// Shifts `addr` to make it aligned with `align` by rounding `addr` to the smallest multiple
    /// of `align` that is larger or equal to `addr`
    fn align_addr(addr: u64, align: u64) -> u64 {
        match addr % align {
            0 => addr,
            rem => addr.checked_add(align).unwrap() - rem,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_addr() {
        assert_eq!(GlobalStateInner::align_addr(37, 4), 40);
        assert_eq!(GlobalStateInner::align_addr(44, 4), 44);
    }
}
