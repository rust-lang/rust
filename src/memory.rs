use rand::rngs::StdRng;

use rustc_mir::interpret::{Pointer, Allocation, AllocationExtra, InterpResult};
use rustc_target::abi::Size;

use crate::{stacked_borrows, intptrcast};
use crate::stacked_borrows::Tag;

#[derive(Clone, Debug)]
pub struct MemoryExtra {
    pub stacked_borrows: stacked_borrows::MemoryExtra,
    pub intptrcast: intptrcast::MemoryExtra,
    /// The random number generator to use if Miri is running in non-deterministic mode and to
    /// enable intptrcast
    pub(crate) rng: Option<StdRng>
}

impl MemoryExtra {
    pub fn with_rng(rng: Option<StdRng>) -> Self {
        MemoryExtra {
            stacked_borrows: Default::default(),
            intptrcast: Default::default(),
            rng,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AllocExtra {
    pub stacked_borrows: stacked_borrows::AllocExtra,
    pub intptrcast: intptrcast::AllocExtra,
}

impl AllocationExtra<Tag> for AllocExtra {
    #[inline(always)]
    fn memory_read<'tcx>(
        alloc: &Allocation<Tag, AllocExtra>,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> InterpResult<'tcx> {
        alloc.extra.stacked_borrows.memory_read(ptr, size)
    }

    #[inline(always)]
    fn memory_written<'tcx>(
        alloc: &mut Allocation<Tag, AllocExtra>,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> InterpResult<'tcx> {
        alloc.extra.stacked_borrows.memory_written(ptr, size)
    }

    #[inline(always)]
    fn memory_deallocated<'tcx>(
        alloc: &mut Allocation<Tag, AllocExtra>,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> InterpResult<'tcx> {
        alloc.extra.stacked_borrows.memory_deallocated(ptr, size)
    }
}
