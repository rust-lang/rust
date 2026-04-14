use crate::BootRuntime;
#[cfg(not(test))]
use crate::memory::kheap::kernel_heap;
#[cfg(not(test))]
use core::alloc::{GlobalAlloc, Layout};

#[cfg(not(test))]
use linked_list_allocator::LockedHeap;

#[cfg(not(test))]
use core::sync::atomic::{AtomicU64, Ordering};

pub static TRACE_ALLOC: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Track the largest allocation seen
#[cfg(not(test))]
static LARGEST_ALLOC: AtomicU64 = AtomicU64::new(0);

/// Count of allocations over 1MB
#[cfg(not(test))]
static LARGE_ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);

/// The inner heap allocator
#[cfg(not(test))]
static INNER_ALLOCATOR: LockedHeap = LockedHeap::empty();

/// Wrapper allocator that logs large allocations
#[cfg(not(test))]
struct TracingAllocator;

#[cfg(not(test))]
unsafe impl GlobalAlloc for TracingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let orig_size = layout.size();
        let orig_align = layout.align();

        // Workaround for linked_list_allocator bug with small leftover holes:
        // By aligning the size to 32 bytes and ensuring a minimum of 32 bytes,
        // we guarantee that any leftover block is at least 32 bytes long
        // (which is > maximum alignment padding + Hole size), preventing it from
        // creating < 24 byte holes that corrupt the free list.
        let align = orig_align.max(8);
        let mut size = orig_size.max(32);
        if size % 32 != 0 {
            size = size + (32 - (size % 32));
        }

        let safe_layout = Layout::from_size_align_unchecked(size, align);

        let irq = crate::irq::irq_disable_erased();
        let ptr = unsafe { INNER_ALLOCATOR.alloc(safe_layout) };
        crate::irq::irq_restore_erased(irq);
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let orig_size = layout.size();
        let orig_align = layout.align();

        let align = orig_align.max(8);
        let mut size = orig_size.max(32);
        if size % 32 != 0 {
            size = size + (32 - (size % 32));
        }

        let safe_layout = Layout::from_size_align_unchecked(size, align);

        let irq = crate::irq::irq_disable_erased();
        unsafe { INNER_ALLOCATOR.dealloc(ptr, safe_layout) }
        crate::irq::irq_restore_erased(irq);
    }
}

#[cfg(not(test))]
#[global_allocator]
static ALLOCATOR: TracingAllocator = TracingAllocator;

#[cfg(not(test))]
pub fn init<R: BootRuntime>(_rt: &R) {
    let mut heap = kernel_heap().lock();
    // Reserve 128MB (32768 pages) for the global heap
    let (base, size) = heap
        .reserve_region::<R>(32768)
        .expect("Failed to reserve kernel heap region");

    unsafe {
        INNER_ALLOCATOR.lock().init(base as *mut u8, size);
    }

    crate::kdebug!("Global allocator initialized (LinkedHeap, 128MB)");
}

/// Get diagnostics about large allocations
#[cfg(not(test))]
pub fn alloc_stats() -> (u64, u64) {
    (
        LARGEST_ALLOC.load(Ordering::Relaxed),
        LARGE_ALLOC_COUNT.load(Ordering::Relaxed),
    )
}

#[cfg(test)]
pub fn init<R: BootRuntime>(_rt: &R) {
    // In tests, we use the system allocator (std), so no manual init needed.
}

#[cfg(test)]
pub fn alloc_stats() -> (u64, u64) {
    (0, 0)
}
