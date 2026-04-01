use std::cell::Cell;

use super::*;

#[test]
fn allocator_param() {
    use crate::alloc::AllocError;

    // Writing a test of integration between third-party
    // allocators and `RawVec` is a little tricky because the `RawVec`
    // API does not expose fallible allocation methods, so we
    // cannot check what happens when allocator is exhausted
    // (beyond detecting a panic).
    //
    // Instead, this just checks that the `RawVec` methods do at
    // least go through the Allocator API when it reserves
    // storage.

    // A dumb allocator that consumes a fixed amount of fuel
    // before allocation attempts start failing.
    struct BoundedAlloc {
        fuel: Cell<usize>,
    }
    unsafe impl Allocator for BoundedAlloc {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            let size = layout.size();
            if size > self.fuel.get() {
                return Err(AllocError);
            }
            match Global.allocate(layout) {
                ok @ Ok(_) => {
                    self.fuel.set(self.fuel.get() - size);
                    ok
                }
                err @ Err(_) => err,
            }
        }
        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            unsafe { Global.deallocate(ptr, layout) }
        }
    }

    let a = BoundedAlloc { fuel: Cell::new(500) };
    let mut v: RawVec<u8, _> = RawVec::with_capacity_in(50, a);
    assert_eq!(v.inner.alloc.fuel.get(), 450);
    v.reserve(50, 150); // (causes a realloc, thus using 50 + 150 = 200 units of fuel)
    assert_eq!(v.inner.alloc.fuel.get(), 250);
}

#[test]
fn reserve_does_not_overallocate() {
    {
        let mut v: RawVec<u32> = RawVec::new();
        // First, `reserve` allocates like `reserve_exact`.
        v.reserve(0, 9);
        assert_eq!(9, v.capacity());
    }

    {
        let mut v: RawVec<u32> = RawVec::new();
        v.reserve(0, 7);
        assert_eq!(7, v.capacity());
        // 97 is more than double of 7, so `reserve` should work
        // like `reserve_exact`.
        v.reserve(7, 90);
        assert_eq!(97, v.capacity());
    }

    {
        let mut v: RawVec<u32> = RawVec::new();
        v.reserve(0, 12);
        assert_eq!(12, v.capacity());
        v.reserve(12, 3);
        // 3 is less than half of 12, so `reserve` must grow
        // exponentially. At the time of writing this test grow
        // factor is 2, so new capacity is 24, however, grow factor
        // of 1.5 is OK too. Hence `>= 18` in assert.
        assert!(v.capacity() >= 12 + 12 / 2);
    }
}

struct ZST;

// A `RawVec` holding zero-sized elements should always look like this.
fn zst_sanity<T>(v: &RawVec<T>) {
    assert_eq!(v.capacity(), usize::MAX);
    assert_eq!(v.ptr(), core::ptr::Unique::<T>::dangling().as_ptr());
    assert_eq!(unsafe { v.inner.current_memory(T::LAYOUT) }, None);
}

#[test]
fn zst() {
    let cap_err = Err(crate::collections::TryReserveErrorKind::CapacityOverflow.into());

    assert_eq!(size_of::<ZST>(), 0);

    // All these different ways of creating the RawVec produce the same thing.

    let v: RawVec<ZST> = RawVec::new();
    zst_sanity(&v);

    let v: RawVec<ZST> = RawVec::with_capacity_in(100, Global);
    zst_sanity(&v);

    let v: RawVec<ZST> = RawVec::with_capacity_in(100, Global);
    zst_sanity(&v);

    let mut v: RawVec<ZST> = RawVec::with_capacity_in(usize::MAX, Global);
    zst_sanity(&v);

    // Check all these operations work as expected with zero-sized elements.

    v.reserve(100, usize::MAX - 100);
    //v.reserve(101, usize::MAX - 100); // panics, in `zst_reserve_panic` below
    zst_sanity(&v);

    v.reserve_exact(100, usize::MAX - 100);
    //v.reserve_exact(101, usize::MAX - 100); // panics, in `zst_reserve_exact_panic` below
    zst_sanity(&v);

    assert_eq!(v.try_reserve(100, usize::MAX - 100), Ok(()));
    assert_eq!(v.try_reserve(101, usize::MAX - 100), cap_err);
    zst_sanity(&v);

    assert_eq!(v.try_reserve_exact(100, usize::MAX - 100), Ok(()));
    assert_eq!(v.try_reserve_exact(101, usize::MAX - 100), cap_err);
    zst_sanity(&v);

    assert_eq!(unsafe { v.inner.grow_amortized(100, usize::MAX - 100, ZST::LAYOUT) }, cap_err);
    assert_eq!(unsafe { v.inner.grow_amortized(101, usize::MAX - 100, ZST::LAYOUT) }, cap_err);
    zst_sanity(&v);

    assert_eq!(unsafe { v.inner.grow_exact(100, usize::MAX - 100, ZST::LAYOUT) }, cap_err);
    assert_eq!(unsafe { v.inner.grow_exact(101, usize::MAX - 100, ZST::LAYOUT) }, cap_err);
    zst_sanity(&v);
}

#[test]
#[should_panic(expected = "capacity overflow")]
fn zst_reserve_panic() {
    let mut v: RawVec<ZST> = RawVec::new();
    zst_sanity(&v);

    v.reserve(101, usize::MAX - 100);
}

#[test]
#[should_panic(expected = "capacity overflow")]
fn zst_reserve_exact_panic() {
    let mut v: RawVec<ZST> = RawVec::new();
    zst_sanity(&v);

    v.reserve_exact(101, usize::MAX - 100);
}

#[test]
fn niches() {
    let baseline = size_of::<RawVec<u8>>();
    assert_eq!(size_of::<Option<RawVec<u8>>>(), baseline);
    assert_eq!(size_of::<Option<Option<RawVec<u8>>>>(), baseline);
    assert_eq!(size_of::<Option<Option<Option<RawVec<u8>>>>>(), baseline);
}
