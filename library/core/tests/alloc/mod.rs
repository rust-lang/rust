use core::alloc::{AllocError, Allocator, Layout};
use core::ptr::NonNull;
use std::{
    collections::HashMap,
    sync::{Mutex, PoisonError},
};

mod const_unchecked_layout;
mod prefix;

#[derive(Default)]
/// Implements `Allocator` and checks it's unsafety conditions.
struct Tracker<A> {
    alloc: A,
    map: Mutex<HashMap<NonNull<u8>, (usize, Layout)>>,
}

impl<A> Tracker<A> {
    fn new(alloc: A) -> Self {
        Self { alloc, map: Default::default() }
    }

    fn after_alloc(&self, layout: Layout, result: Result<NonNull<[u8]>, AllocError>) {
        if let Ok(ptr) = result {
            self.map
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(ptr.as_non_null_ptr(), (ptr.len(), layout));
        }
    }

    fn before_dealloc(&self, ptr: NonNull<u8>, layout: Layout) {
        let lock = self.map.lock().unwrap_or_else(PoisonError::into_inner);
        let (size, old_layout) = lock
            .get(&ptr)
            .expect("`ptr` must denote a block of memory currently allocated via this allocator");
        assert_eq!(
            layout.align(),
            old_layout.align(),
            "`layout` must fit that block of memory. Expected alignment of {}, got {}",
            old_layout.align(),
            layout.align()
        );
        if layout.size() < old_layout.size() || layout.size() > *size {
            if *size == old_layout.size() {
                panic!(
                    "`layout` must fit that block of memory. Expected size of {}, got {}",
                    old_layout.size(),
                    layout.size()
                )
            } else {
                panic!(
                    "`layout` must fit that block of memory. Expected size between {}..={}, \
                        got {}",
                    old_layout.size(),
                    size,
                    layout.size()
                )
            }
        }
    }

    fn after_dealloc(&self, ptr: NonNull<u8>, _layout: Layout) {
        self.map.lock().unwrap_or_else(PoisonError::into_inner).remove(&ptr);
    }

    fn before_grow(&self, ptr: NonNull<u8>, old_layout: Layout, new_layout: Layout) {
        assert!(
            new_layout.size() >= old_layout.size(),
            "`new_layout.size()` must be greater than or equal to `old_layout.size()`, expected {} >= {}",
            new_layout.size(),
            old_layout.size()
        );
        self.before_dealloc(ptr, old_layout)
    }

    #[track_caller]
    fn after_grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
        result: Result<NonNull<[u8]>, AllocError>,
    ) {
        if result.is_ok() {
            self.after_dealloc(ptr, old_layout);
            self.after_alloc(new_layout, result);
        }
    }

    fn before_shrink(&self, ptr: NonNull<u8>, old_layout: Layout, new_layout: Layout) {
        assert!(
            new_layout.size() <= old_layout.size(),
            "`new_layout.size()` must be smaller than or equal to `old_layout.size()`, expected {} >= {}",
            new_layout.size(),
            old_layout.size()
        );
        self.before_dealloc(ptr, old_layout)
    }

    #[track_caller]
    fn after_shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
        result: Result<NonNull<[u8]>, AllocError>,
    ) {
        if result.is_ok() {
            self.after_dealloc(ptr, old_layout);
            self.after_alloc(new_layout, result);
        }
    }
}

unsafe impl<A: Allocator> Allocator for Tracker<A> {
    #[track_caller]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, std::alloc::AllocError> {
        let result = self.alloc.allocate(layout);
        self.after_alloc(layout, result);
        result
    }

    #[track_caller]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, std::alloc::AllocError> {
        let result = self.alloc.allocate_zeroed(layout);
        self.after_alloc(layout, result);
        result
    }

    #[track_caller]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.before_dealloc(ptr, layout);
        unsafe { self.alloc.deallocate(ptr, layout) }
        self.after_dealloc(ptr, layout);
    }

    #[track_caller]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, std::alloc::AllocError> {
        self.before_grow(ptr, old_layout, new_layout);
        let result = unsafe { self.alloc.grow(ptr, old_layout, new_layout) };
        self.after_grow(ptr, old_layout, new_layout, result);
        result
    }

    #[track_caller]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, std::alloc::AllocError> {
        self.before_grow(ptr, old_layout, new_layout);
        let result = unsafe { self.alloc.grow_zeroed(ptr, old_layout, new_layout) };
        self.after_grow(ptr, old_layout, new_layout, result);
        result
    }

    #[track_caller]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, std::alloc::AllocError> {
        self.before_shrink(ptr, old_layout, new_layout);
        let result = unsafe { self.alloc.shrink(ptr, old_layout, new_layout) };
        self.after_shrink(ptr, old_layout, new_layout, result);
        result
    }
}

impl<A> Drop for Tracker<A> {
    fn drop(&mut self) {
        let lock = self.map.lock().unwrap_or_else(PoisonError::into_inner);
        if !lock.is_empty() {
            panic!("Missing deallocations {:#?}", lock);
        }
    }
}
