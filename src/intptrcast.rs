use std::cell::RefCell;
use std::collections::hash_map::Entry;

use log::trace;
use rand::Rng;

use rustc_data_structures::fx::FxHashMap;
use rustc_target::abi::{HasDataLayout, Size};

use crate::*;

pub type MemoryExtra = RefCell<GlobalState>;

#[derive(Clone, Debug)]
pub struct GlobalState {
    /// This is used as a map between the address of each allocation and its `AllocId`.
    /// It is always sorted
    pub int_to_ptr_map: Vec<(u64, AllocId)>,
    /// The base address for each allocation.  We cannot put that into
    /// `AllocExtra` because function pointers also have a base address, and
    /// they do not have an `AllocExtra`.
    /// This is the inverse of `int_to_ptr_map`.
    pub base_addr: FxHashMap<AllocId, u64>,
    /// This is used as a memory address when a new pointer is casted to an integer. It
    /// is always larger than any address that was previously made part of a block.
    pub next_base_addr: u64,
}

impl Default for GlobalState {
    fn default() -> Self {
        GlobalState {
            int_to_ptr_map: Vec::default(),
            base_addr: FxHashMap::default(),
            next_base_addr: STACK_ADDR,
        }
    }
}

impl<'mir, 'tcx> GlobalState {
    pub fn ptr_from_addr(
        addr: u64,
        memory: &Memory<'mir, 'tcx, Evaluator<'mir, 'tcx>>,
    ) -> Pointer<Option<Tag>> {
        trace!("Casting 0x{:x} to a pointer", addr);
        let global_state = memory.extra.intptrcast.borrow();
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
                    <= memory.get_size_and_align(alloc_id, AllocCheck::MaybeDead).unwrap().0.bytes()
                {
                    Some(alloc_id)
                } else {
                    None
                }
            }
        };
        // Pointers created from integers are untagged.
        Pointer::new(
            alloc_id.map(|alloc_id| Tag { alloc_id, sb: SbTag::Untagged }),
            Size::from_bytes(addr),
        )
    }

    fn alloc_base_addr(
        memory: &Memory<'mir, 'tcx, Evaluator<'mir, 'tcx>>,
        alloc_id: AllocId,
    ) -> u64 {
        let mut global_state = memory.extra.intptrcast.borrow_mut();
        let global_state = &mut *global_state;

        match global_state.base_addr.entry(alloc_id) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                // There is nothing wrong with a raw pointer being cast to an integer only after
                // it became dangling.  Hence `MaybeDead`.
                let (size, align) =
                    memory.get_size_and_align(alloc_id, AllocCheck::MaybeDead).unwrap();

                // This allocation does not have a base address yet, pick one.
                // Leave some space to the previous allocation, to give it some chance to be less aligned.
                let slack = {
                    let mut rng = memory.extra.rng.borrow_mut();
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
    pub fn rel_ptr_to_addr(
        memory: &Memory<'mir, 'tcx, Evaluator<'mir, 'tcx>>,
        ptr: Pointer<AllocId>,
    ) -> u64 {
        let (alloc_id, offset) = ptr.into_parts(); // offset is relative
        let base_addr = GlobalState::alloc_base_addr(memory, alloc_id);

        // Add offset with the right kind of pointer-overflowing arithmetic.
        let dl = memory.data_layout();
        dl.overflowing_offset(base_addr, offset.bytes()).0
    }

    pub fn abs_ptr_to_rel(
        memory: &Memory<'mir, 'tcx, Evaluator<'mir, 'tcx>>,
        ptr: Pointer<Tag>,
    ) -> Size {
        let (tag, addr) = ptr.into_parts(); // addr is absolute
        let base_addr = GlobalState::alloc_base_addr(memory, tag.alloc_id);

        // Wrapping "addr - base_addr"
        let dl = memory.data_layout();
        let neg_base_addr = (base_addr as i64).wrapping_neg();
        Size::from_bytes(dl.overflowing_signed_offset(addr.bytes(), neg_base_addr).0)
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
        assert_eq!(GlobalState::align_addr(37, 4), 40);
        assert_eq!(GlobalState::align_addr(44, 4), 44);
    }
}
