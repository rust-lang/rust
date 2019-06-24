use std::cell::{Cell, RefCell};

use rustc::mir::interpret::{AllocId, Pointer, InterpResult};
use rustc_mir::interpret::Memory;
use rustc_target::abi::Size;

use crate::stacked_borrows::Tag;
use crate::Evaluator;

pub type MemoryExtra = RefCell<GlobalState>;

#[derive(Clone, Debug, Default)]
pub struct AllocExtra {
    base_addr: Cell<Option<u64>>
}

#[derive(Clone, Debug)]
pub struct GlobalState {
    /// This is used as a map between the address of each allocation and its `AllocId`.
    /// It is always sorted
    pub int_to_ptr_map: Vec<(u64, AllocId)>,
    /// This is used as a memory address when a new pointer is casted to an integer. It
    /// is always larger than any address that was previously made part of a block.
    pub next_base_addr: u64,
}

impl Default for GlobalState {
    // FIXME: Query the page size in the future
    fn default() -> Self {
        GlobalState {
            int_to_ptr_map: Vec::default(),
            next_base_addr: 2u64.pow(16)
        }
    }
}

impl<'mir, 'tcx> GlobalState {
    pub fn int_to_ptr(
        base_addr: u64,
        memory: &Memory<'mir, 'tcx, Evaluator<'tcx>>,
    ) -> InterpResult<'tcx, Pointer<Tag>> {
        let global_state = memory.extra.intptrcast.borrow();
        
        match global_state.int_to_ptr_map.binary_search_by_key(&base_addr, |(addr, _)| *addr) {
            Ok(pos) => {
                let (_, alloc_id) = global_state.int_to_ptr_map[pos];
                // `base_addr` is the starting address for an allocation, the offset should be
                // zero. The pointer is untagged because it was created from a cast
                Ok(Pointer::new_with_tag(alloc_id, Size::from_bytes(0), Tag::Untagged))
            },
            Err(0) => err!(DanglingPointerDeref), 
            Err(pos) => {
                // This is the gargest of the adresses smaller than `base_addr`,
                // i.e. the greatest lower bound (glb)
                let (glb, alloc_id) = global_state.int_to_ptr_map[pos - 1];
                // This never overflows because `base_addr >= glb`
                let offset = base_addr - glb;
                // If the offset exceeds the size of the allocation, this access is illegal
                if offset <= memory.get(alloc_id)?.bytes.len() as u64 {
                    // This pointer is untagged because it was created from a cast
                    Ok(Pointer::new_with_tag(alloc_id, Size::from_bytes(offset), Tag::Untagged))
                } else {
                    err!(DanglingPointerDeref)
                } 
            }
        }
    }

    pub fn ptr_to_int(
        ptr: Pointer<Tag>,
        memory: &Memory<'mir, 'tcx, Evaluator<'tcx>>,
    ) -> InterpResult<'tcx, u64> {
        let mut global_state = memory.extra.intptrcast.borrow_mut();

        let alloc = memory.get(ptr.alloc_id)?;

        let base_addr = match alloc.extra.intptrcast.base_addr.get() { 
            Some(base_addr) => base_addr,
            None => {
                let base_addr = global_state.next_base_addr;
                global_state.next_base_addr += alloc.bytes.len() as u64;

                alloc.extra.intptrcast.base_addr.set(Some(base_addr));

                let elem = (base_addr, ptr.alloc_id);

                // Given that `next_base_addr` increases in each allocation, pushing the
                // corresponding tuple keeps `int_to_ptr_map` sorted
                global_state.int_to_ptr_map.push(elem); 

                base_addr
            }
        };

        Ok(base_addr + ptr.offset.bytes())
    }
}
