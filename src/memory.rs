use std::cell::RefCell;

use rustc_mir::interpret::{Pointer, Allocation, AllocationExtra, InterpResult};
use rustc_target::abi::Size;

use crate::{stacked_borrows, intptrcast};
use crate::stacked_borrows::{Tag, AccessKind};

#[derive(Default, Clone, Debug)]
pub struct MemoryState {
    pub stacked: stacked_borrows::MemoryState,
    pub intptrcast: intptrcast::MemoryState,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone,)]
pub struct AllocExtra {
    pub stacks: stacked_borrows::Stacks,
    pub base_addr: RefCell<Option<u64>>,
}

impl AllocationExtra<Tag> for AllocExtra {
    #[inline(always)]
    fn memory_read<'tcx>(
        alloc: &Allocation<Tag, AllocExtra>,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> InterpResult<'tcx> {
        trace!("read access with tag {:?}: {:?}, size {}", ptr.tag, ptr.erase_tag(), size.bytes());
        alloc.extra.stacks.for_each(ptr, size, |stack, global| {
            stack.access(AccessKind::Read, ptr.tag, global)?;
            Ok(())
        })
    }

    #[inline(always)]
    fn memory_written<'tcx>(
        alloc: &mut Allocation<Tag, AllocExtra>,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> InterpResult<'tcx> {
        trace!("write access with tag {:?}: {:?}, size {}", ptr.tag, ptr.erase_tag(), size.bytes());
        alloc.extra.stacks.for_each(ptr, size, |stack, global| {
            stack.access(AccessKind::Write, ptr.tag, global)?;
            Ok(())
        })
    }

    #[inline(always)]
    fn memory_deallocated<'tcx>(
        alloc: &mut Allocation<Tag, AllocExtra>,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> InterpResult<'tcx> {
        trace!("deallocation with tag {:?}: {:?}, size {}", ptr.tag, ptr.erase_tag(), size.bytes());
        alloc.extra.stacks.for_each(ptr, size, |stack, global| {
            stack.dealloc(ptr.tag, global)
        })
    }
}
