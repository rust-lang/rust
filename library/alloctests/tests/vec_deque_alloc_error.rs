#![feature(alloc_error_hook, allocator_api)]

use std::alloc::{AllocError, Allocator, Layout, System, set_alloc_error_hook};
use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr::NonNull;

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_shrink_to_unwind() {
    // This tests that `shrink_to` leaves the deque in a consistent state when
    // the call to `RawVec::shrink_to_fit` unwinds. The code is adapted from #123369
    // but changed to hopefully not have any UB even if the test fails.

    struct BadAlloc;

    unsafe impl Allocator for BadAlloc {
        fn allocate(&self, l: Layout) -> Result<NonNull<[u8]>, AllocError> {
            // We allocate zeroed here so that the whole buffer of the deque
            // is always initialized. That way, even if the deque is left in
            // an inconsistent state, no uninitialized memory should be accessed.
            System.allocate_zeroed(l)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            unsafe { System.deallocate(ptr, layout) }
        }

        unsafe fn shrink(
            &self,
            _ptr: NonNull<u8>,
            _old_layout: Layout,
            _new_layout: Layout,
        ) -> Result<NonNull<[u8]>, AllocError> {
            Err(AllocError)
        }
    }

    set_alloc_error_hook(|_| panic!("alloc error"));

    let mut v = VecDeque::with_capacity_in(15, BadAlloc);
    v.push_back(1);
    v.push_front(2);
    // This should unwind because it calls `BadAlloc::shrink` and then `handle_alloc_error` which unwinds.
    assert!(catch_unwind(AssertUnwindSafe(|| v.shrink_to_fit())).is_err());
    // This should only pass if the deque is left in a consistent state.
    assert_eq!(v, [2, 1]);
}
