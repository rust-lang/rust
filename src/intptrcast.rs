use std::cell::RefCell;
use std::collections::{HashMap, hash_map::Entry};
use std::cmp::max;

use rand::Rng;

use rustc::ty::layout::HasDataLayout;
use rustc_mir::interpret::{AllocId, Pointer, InterpResult, Memory, AllocCheck, PointerArithmetic};
use rustc_target::abi::Size;

use crate::{Evaluator, Tag, STACK_ADDR};

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
    pub base_addr: HashMap<AllocId, u64>,
    /// This is used as a memory address when a new pointer is casted to an integer. It
    /// is always larger than any address that was previously made part of a block.
    pub next_base_addr: u64,
}

impl Default for GlobalState {
    fn default() -> Self {
        GlobalState {
            int_to_ptr_map: Vec::default(),
            base_addr: HashMap::default(),
            next_base_addr: STACK_ADDR,
        }
    }
}

impl<'mir, 'tcx> GlobalState {
    pub fn int_to_ptr(
        int: u64,
        memory: &Memory<'mir, 'tcx, Evaluator<'tcx>>,
    ) -> InterpResult<'tcx, Pointer<Tag>> {
        if int == 0 {
            throw_unsup!(InvalidNullPointerUsage);
        }

        let global_state = memory.extra.intptrcast.borrow();

        Ok(match global_state.int_to_ptr_map.binary_search_by_key(&int, |(addr, _)| *addr) {
            Ok(pos) => {
                let (_, alloc_id) = global_state.int_to_ptr_map[pos];
                // `int` is equal to the starting address for an allocation, the offset should be
                // zero. The pointer is untagged because it was created from a cast
                Pointer::new_with_tag(alloc_id, Size::from_bytes(0), Tag::Untagged)
            },
            Err(0) => throw_unsup!(DanglingPointerDeref),
            Err(pos) => {
                // This is the largest of the adresses smaller than `int`,
                // i.e. the greatest lower bound (glb)
                let (glb, alloc_id) = global_state.int_to_ptr_map[pos - 1];
                // This never overflows because `int >= glb`
                let offset = int - glb;
                // If the offset exceeds the size of the allocation, this access is illegal
                if offset <= memory.get_size_and_align(alloc_id, AllocCheck::MaybeDead)?.0.bytes() {
                    // This pointer is untagged because it was created from a cast
                    Pointer::new_with_tag(alloc_id, Size::from_bytes(offset), Tag::Untagged)
                } else {
                    throw_unsup!(DanglingPointerDeref)
                }
            }
        })
    }

    pub fn ptr_to_int(
        ptr: Pointer<Tag>,
        memory: &Memory<'mir, 'tcx, Evaluator<'tcx>>,
    ) -> InterpResult<'tcx, u64> {
        let mut global_state = memory.extra.intptrcast.borrow_mut();
        let global_state = &mut *global_state;

        // There is nothing wrong with a raw pointer being cast to an integer only after
        // it became dangling.  Hence `MaybeDead`.
        let (size, align) = memory.get_size_and_align(ptr.alloc_id, AllocCheck::MaybeDead)?;

        let base_addr = match global_state.base_addr.entry(ptr.alloc_id) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                // This allocation does not have a base address yet, pick one.
                // Leave some space to the previous allocation, to give it some chance to be less aligned.
                let slack = {
                    let mut rng = memory.extra.rng.borrow_mut();
                    // This means that `(global_state.next_base_addr + slack) % 16` is uniformly distributed.
                    rng.gen_range(0, 16)
                };
                // From next_base_addr + slack, round up to adjust for alignment.
                let base_addr = global_state.next_base_addr.checked_add(slack).unwrap();
                let base_addr = Self::align_addr(base_addr, align.bytes());
                entry.insert(base_addr);
                trace!(
                    "Assigning base address {:#x} to allocation {:?} (slack: {}, align: {})",
                    base_addr, ptr.alloc_id, slack, align.bytes(),
                );

                // Remember next base address.  If this allocation is zero-sized, leave a gap
                // of at least 1 to avoid two allocations having the same base address.
                global_state.next_base_addr = base_addr.checked_add(max(size.bytes(), 1)).unwrap();
                // Given that `next_base_addr` increases in each allocation, pushing the
                // corresponding tuple keeps `int_to_ptr_map` sorted
                global_state.int_to_ptr_map.push((base_addr, ptr.alloc_id));

                base_addr
            }
        };

        debug_assert_eq!(base_addr % align.bytes(), 0); // sanity check
        // Add offset with the right kind of pointer-overflowing arithmetic.
        let dl = memory.data_layout();
        Ok(dl.overflowing_offset(base_addr, ptr.offset.bytes()).0)
    }

    /// Shifts `addr` to make it aligned with `align` by rounding `addr` to the smallest multiple
    /// of `align` that is larger or equal to `addr`
    fn align_addr(addr: u64, align: u64) -> u64 {
        match addr % align {
            0 => addr,
            rem => addr.checked_add(align).unwrap() - rem
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
