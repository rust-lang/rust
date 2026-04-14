use core::alloc::{GlobalAlloc, Layout};
use core::ptr;

use linked_list_allocator::LockedHeap;

pub struct VmHeapAllocator {
    heap: LockedHeap,
}

impl VmHeapAllocator {
    pub const fn new() -> Self {
        Self {
            heap: LockedHeap::empty(),
        }
    }

    pub unsafe fn init(&self, heap_start: usize, heap_size: usize) {
        self.heap
            .lock()
            .init(heap_start as *mut u8, heap_size);
    }

    pub unsafe fn extend(&self, by: usize) {
        self.heap.lock().extend(by);
    }
}

#[global_allocator]
static ALLOCATOR: VmHeapAllocator = VmHeapAllocator::new();

pub(crate) fn init_heap(heap_start: usize, heap_size: usize) {
    unsafe {
        ALLOCATOR.init(heap_start, heap_size);
    }
}

pub(crate) fn extend_heap(by: usize) {
    unsafe {
        ALLOCATOR.extend(by);
    }
}

unsafe impl GlobalAlloc for VmHeapAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() == 0 {
            return layout.dangling().as_ptr();
        }

        let ptr = GlobalAlloc::alloc(&self.heap, layout);
        if !ptr.is_null() {
            return ptr;
        }

        if crate::heap::grow_heap(layout.size()).is_ok() {
            return GlobalAlloc::alloc(&self.heap, layout);
        }

        ptr::null_mut()
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if layout.size() == 0 {
            return;
        }
        GlobalAlloc::dealloc(&self.heap, ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if ptr.is_null() {
            return self.alloc(Layout::from_size_align_unchecked(new_size, layout.align()));
        }
        if new_size == 0 {
            self.dealloc(ptr, layout);
            return ptr::null_mut();
        }

        let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
        let new_ptr = self.alloc(new_layout);
        if new_ptr.is_null() {
            return ptr::null_mut();
        }

        let copy_len = core::cmp::min(layout.size(), new_size);
        ptr::copy_nonoverlapping(ptr, new_ptr, copy_len);
        self.dealloc(ptr, layout);
        new_ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = self.alloc(layout);
        if ptr.is_null() {
            return ptr;
        }
        ptr::write_bytes(ptr, 0, layout.size());
        ptr
    }
}
