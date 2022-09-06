use super::*;
use std::cell::Cell;

// FIXME: calculations here are not as easy as they were before
#[ignore]
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
    assert_eq!(v.alloc.fuel.get(), 450);
    v.reserve(50, 150); // (causes a realloc, thus using 50 + 150 = 200 units of fuel)
    assert_eq!(v.alloc.fuel.get(), 250);
}

#[test]
fn grow_amortized_power_of_two_bins() {
    // capacity is set to fit `2^n` (bin_size) minus `usize`-sized overhead
    fn cap_for<T>(bin_size: usize) -> usize {
        (bin_size - mem::size_of::<usize>()) / mem::size_of::<T>()
    }

    {
        let mut v: RawVec<u32> = RawVec::new();
        v.reserve(0, 9);
        assert_eq!(cap_for::<u32>(64), v.capacity());
    }

    {
        let mut v: RawVec<u32> = RawVec::new();
        v.reserve(0, 7);
        assert_eq!(cap_for::<u32>(64), v.capacity());
        v.reserve(7, 90);
        // above the limit where we still try to align to bin size
        assert_eq!(128, v.capacity());
    }

    {
        let mut v: RawVec<u32> = RawVec::new();
        v.reserve(0, 14);
        assert_eq!(cap_for::<u32>(64), v.capacity());
        v.reserve(14, 1);
        assert_eq!(cap_for::<u32>(128), v.capacity());
    }

    {
        let mut v: RawVec<u8> = RawVec::new();
        v.reserve(0, 1);
        assert_eq!(cap_for::<u8>(16), v.capacity());
        v.reserve(0, 7);
        assert_eq!(cap_for::<u8>(16), v.capacity());
        v.reserve(0, 8);
        assert_eq!(cap_for::<u8>(16), v.capacity());
        v.reserve(0, 9);
        assert_eq!(cap_for::<u8>(32), v.capacity());
        v.reserve(8, 1);
        assert_eq!(cap_for::<u8>(32), v.capacity());
    }

    {
        let mut v: RawVec<u8> = RawVec::new();
        v.reserve_exact(0, 6);
        assert_eq!(6, v.capacity());
        v.reserve(0, 8);
        // increase all the way to 32 (instead of just 32), due to minimum capacity increase
        assert_eq!(cap_for::<u8>(32), v.capacity());
    }

    {
        let mut v: RawVec<[u8; 5]> = RawVec::new();
        v.reserve(0, 1);
        assert_eq!(cap_for::<[u8; 5]>(16), v.capacity());
        v.reserve(0, 2);
        assert_eq!(cap_for::<[u8; 5]>(32), v.capacity());
        v.reserve(0, 3);
        assert_eq!(cap_for::<[u8; 5]>(32), v.capacity());
    }
}

struct ZST;

// A `RawVec` holding zero-sized elements should always look like this.
fn zst_sanity<T>(v: &RawVec<T>) {
    assert_eq!(v.capacity(), usize::MAX);
    assert_eq!(v.ptr(), core::ptr::Unique::<T>::dangling().as_ptr());
    assert_eq!(v.current_memory(), None);
}

#[test]
fn zst() {
    let cap_err = Err(crate::collections::TryReserveErrorKind::CapacityOverflow.into());

    assert_eq!(std::mem::size_of::<ZST>(), 0);

    // All these different ways of creating the RawVec produce the same thing.

    let v: RawVec<ZST> = RawVec::new();
    zst_sanity(&v);

    let v: RawVec<ZST> = RawVec::with_capacity_in(100, Global);
    zst_sanity(&v);

    let v: RawVec<ZST> = RawVec::with_capacity_in(100, Global);
    zst_sanity(&v);

    let v: RawVec<ZST> = RawVec::allocate_in(0, AllocInit::Uninitialized, Global);
    zst_sanity(&v);

    let v: RawVec<ZST> = RawVec::allocate_in(100, AllocInit::Uninitialized, Global);
    zst_sanity(&v);

    let mut v: RawVec<ZST> = RawVec::allocate_in(usize::MAX, AllocInit::Uninitialized, Global);
    zst_sanity(&v);

    // Check all these operations work as expected with zero-sized elements.

    assert!(!v.needs_to_grow(100, usize::MAX - 100));
    assert!(v.needs_to_grow(101, usize::MAX - 100));
    zst_sanity(&v);

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

    assert_eq!(v.grow_amortized(100, usize::MAX - 100), cap_err);
    assert_eq!(v.grow_amortized(101, usize::MAX - 100), cap_err);
    zst_sanity(&v);

    assert_eq!(v.grow_exact(100, usize::MAX - 100), cap_err);
    assert_eq!(v.grow_exact(101, usize::MAX - 100), cap_err);
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
