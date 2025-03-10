use core::cell::SyncUnsafeCell;
use core::sync::atomic::AtomicUsize;

use crate::BsanAllocator;
#[cfg(test)]
use crate::TEST_ALLOC;

#[derive(Debug)]
pub struct GlobalContext {
    allocator: BsanAllocator,
    next_alloc_id: AtomicUsize,

    // pointer to l1 table
    shadow_heap: ShadowMap<crate::Provenance>,
}

impl GlobalContext {
    fn new(allocator: BsanAllocator) -> Self {

        //
        Self { allocator, next_alloc_id: AtomicUsize::new(1) }

        // TODO: Before returning, make sure to create and init a shadowheap object, including the L1 table using MMMAP
        // TODO: Follow line 189 on the softboundsCETS.cpp to implement it, with the same flags and asserts

    }
}

pub static GLOBAL_CTX: SyncUnsafeCell<Option<GlobalContext>> = SyncUnsafeCell::new(None);

pub unsafe fn init_global_ctx(alloc: BsanAllocator) {
    *GLOBAL_CTX.get() = Some(GlobalContext::new(alloc));
}

#[inline]
pub unsafe fn global_ctx() -> &'static GlobalContext {
    (&(*GLOBAL_CTX.get())).as_ref().unwrap_unchecked()
}
