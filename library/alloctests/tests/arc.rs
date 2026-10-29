use std::any::Any;
use std::cell::{Cell, RefCell};
use std::iter::TrustedLen;
use std::sync::{Arc, UniqueArc, Weak};

struct ArcNDrop<'a>(&'a Cell<usize>);

impl Drop for ArcNDrop<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

#[test]
fn n_new() {
    for n in [1, 2, 17] {
        let mut iter = Arc::new_n([1, 2, 3], n);
        assert_eq!(iter.len(), n);
        assert_eq!(iter.size_hint(), (n, Some(n)));
        let first = iter.next().unwrap();
        assert_eq!(Arc::strong_count(&first), n);
        for remaining in (0..n - 1).rev() {
            let next = iter.next().unwrap();
            assert!(Arc::ptr_eq(&first, &next));
            assert_eq!(*next, [1, 2, 3]);
            assert_eq!(iter.len(), remaining);
            assert_eq!(Arc::strong_count(&first), remaining + 2);
        }
        assert!(iter.next().is_none());
        assert!(iter.next().is_none());
        assert_eq!(Arc::strong_count(&first), 1);
    }
}

#[test]
fn n_try_new() {
    let mut iter = Arc::try_new_n(String::from("value"), 3).unwrap();
    let first = iter.next().unwrap();
    assert_eq!(Arc::strong_count(&first), 3);
    let rest: Vec<_> = iter.collect();
    assert_eq!(rest.len(), 2);
    assert!(rest.iter().all(|arc| Arc::ptr_eq(&first, arc)));
    drop(rest);
    assert_eq!(Arc::strong_count(&first), 1);
}

#[test]
fn n_new_zero() {
    let drops = Cell::new(0);
    let mut iter = Arc::new_n(ArcNDrop(&drops), 0);
    assert_eq!(drops.get(), 1);
    assert_eq!(iter.len(), 0);
    assert!(iter.next().is_none());
    drop(iter);
    assert_eq!(drops.get(), 1);

    let mut iter = Arc::try_new_n(ArcNDrop(&drops), 0).unwrap();
    assert_eq!(drops.get(), 2);
    assert_eq!(iter.len(), 0);
    assert!(iter.next().is_none());
    drop(iter);
    assert_eq!(drops.get(), 2);
}

#[test]
fn n_new_drop_unconsumed() {
    let drops = Cell::new(0);
    drop(Arc::new_n(ArcNDrop(&drops), 8));
    assert_eq!(drops.get(), 1);
    drop(Arc::try_new_n(ArcNDrop(&drops), 8).unwrap());
    assert_eq!(drops.get(), 2);
}

#[test]
fn n_new_drop_partial() {
    let drops = Cell::new(0);
    let mut iter = Arc::new_n(ArcNDrop(&drops), 8);
    let first = iter.next().unwrap();
    let weak = Arc::downgrade(&first);
    drop(iter);
    assert_eq!(Arc::strong_count(&first), 1);
    assert_eq!(drops.get(), 0);
    drop(first);
    assert_eq!(drops.get(), 1);
    assert!(weak.upgrade().is_none());
}

#[test]
fn n_new_iterator_keeps_allocation_alive() {
    let drops = Cell::new(0);
    let mut iter = Arc::new_n(ArcNDrop(&drops), 8);
    let first = iter.next().unwrap();
    let weak = Arc::downgrade(&first);
    drop(first);
    assert_eq!(drops.get(), 0);
    assert_eq!(weak.strong_count(), 7);
    drop(iter);
    assert_eq!(drops.get(), 1);
    assert!(weak.upgrade().is_none());
}

#[test]
fn n_clone() {
    let original = Arc::new([1, 2, 3]);
    let mut iter = Arc::clone_n(&original, 3);
    assert_eq!(Arc::strong_count(&original), 4);
    assert_eq!(iter.len(), 3);
    let first = iter.next().unwrap();
    assert_eq!(Arc::strong_count(&original), 4);
    assert!(Arc::ptr_eq(&original, &first));
    drop(iter);
    assert_eq!(Arc::strong_count(&original), 2);
    drop(first);
    assert_eq!(Arc::strong_count(&original), 1);
}

#[test]
fn n_clone_outlives_source() {
    let drops = Cell::new(0);
    let original = Arc::new(ArcNDrop(&drops));
    let mut iter = Arc::clone_n(&original, 2);
    drop(original);
    let first = iter.next().unwrap();
    assert_eq!(Arc::strong_count(&first), 2);
    drop(iter);
    assert_eq!(drops.get(), 0);
    drop(first);
    assert_eq!(drops.get(), 1);
}

#[test]
fn n_clone_zero() {
    let original = Arc::new(1);
    let mut iter = Arc::clone_n(&original, 0);
    assert_eq!(Arc::strong_count(&original), 1);
    assert_eq!(iter.size_hint(), (0, Some(0)));
    assert!(iter.next().is_none());
    assert!(iter.next().is_none());
    drop(iter);
    assert_eq!(Arc::strong_count(&original), 1);
}

#[test]
fn n_clone_unsized() {
    let original: Arc<[i32]> = Arc::from([1, 2, 3]);
    let clones: Vec<_> = Arc::clone_n(&original, 4).collect();
    assert_eq!(Arc::strong_count(&original), 5);
    assert!(clones.iter().all(|arc| Arc::ptr_eq(&original, arc)));
    drop(clones);

    let original: Arc<dyn Any> = Arc::new(42);
    let clone = Arc::clone_n(&original, 1).next().unwrap();
    assert!(Arc::ptr_eq(&original, &clone));
    assert_eq!(clone.downcast_ref::<i32>(), Some(&42));
}

#[test]
fn n_iter_send_sync_fused() {
    fn check<I: Send + Sync + std::iter::FusedIterator>(_: &I) {}
    let iter = Arc::new_n(42, 4);
    check(&iter);
    let iter = std::thread::spawn(move || iter).join().unwrap();
    assert_eq!(iter.count(), 4);
    let original = Arc::new(42);
    let iter = Arc::clone_n(&original, 4);
    check(&iter);
    let clones = std::thread::spawn(move || iter.collect::<Vec<_>>()).join().unwrap();
    assert_eq!(Arc::strong_count(&original), 5);
    drop(clones);
}

#[test]
fn n_iter_adapters() {
    let original = Arc::new(42);
    let last = Arc::clone_n(&original, 8).nth(7).unwrap();
    assert_eq!(Arc::strong_count(&original), 2);
    drop(last);
    let first: Vec<_> = Arc::clone_n(&original, 8).take(2).collect();
    assert_eq!(Arc::strong_count(&original), 3);
    drop(first);
    assert_eq!(Arc::clone_n(&original, 8).count(), 8);
    assert_eq!(Arc::strong_count(&original), 1);
}

#[test]
fn n_clone_batch_boundaries() {
    let max_batch = 1usize << (usize::BITS / 2);
    let original = Arc::new(42);
    for n in [max_batch - 1, max_batch, max_batch + 1] {
        let mut iter = Arc::clone_n(&original, n);
        assert_eq!(iter.len(), n);
        assert_eq!(Arc::strong_count(&original), n + 1);
        let first = iter.next().unwrap();
        assert!(Arc::ptr_eq(&original, &first));
        assert_eq!(Arc::strong_count(&original), n + 1);
        drop(iter);
        assert_eq!(Arc::strong_count(&original), 2);
        drop(first);
        assert_eq!(Arc::strong_count(&original), 1);
    }
}

#[test]
fn n_clone_concurrent_batches() {
    let n = 1usize << (usize::BITS / 2);
    let original = Arc::new(42);
    let barrier = std::sync::Barrier::new(8);
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                let iter = Arc::clone_n(&original, n);
                barrier.wait();
                assert_eq!(Arc::strong_count(&original), 8 * n + 1);
                barrier.wait();
                drop(iter);
            });
        }
    });
    assert_eq!(Arc::strong_count(&original), 1);
}

#[test]
#[cfg(panic = "unwind")]
fn n_new_overflow() {
    let drops = Cell::new(0);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Arc::new_n(ArcNDrop(&drops), isize::MAX as usize + 1)
        }))
        .is_err()
    );
    assert_eq!(drops.get(), 1);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Arc::try_new_n(ArcNDrop(&drops), usize::MAX)
        }))
        .is_err()
    );
    assert_eq!(drops.get(), 2);
}

#[test]
#[cfg(panic = "unwind")]
fn n_clone_overflow() {
    let slow_batch = (1usize << (usize::BITS / 2)) + 1;
    let original = Arc::new(());
    let reserved = Arc::clone_n(&original, isize::MAX as usize - 1);
    assert_eq!(Arc::strong_count(&original), isize::MAX as usize);
    assert!(std::panic::catch_unwind(|| Arc::clone_n(&original, slow_batch)).is_err());
    assert!(std::panic::catch_unwind(|| Arc::clone_n(&original, usize::MAX)).is_err());
    assert_eq!(Arc::strong_count(&original), isize::MAX as usize);
    drop(Arc::clone_n(&original, 0));
    drop(reserved);
    assert_eq!(Arc::strong_count(&original), 1);
}

#[test]
#[cfg(panic = "unwind")]
fn n_clone_concurrent_overflow() {
    let slow_batch = (1usize << (usize::BITS / 2)) + 1;
    let original = Arc::new(());
    let reserved = Arc::clone_n(&original, isize::MAX as usize - slow_batch - 1);
    let barrier = std::sync::Barrier::new(8);
    let successes = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                barrier.wait();
                let result = std::panic::catch_unwind(|| Arc::clone_n(&original, slow_batch));
                if result.is_ok() {
                    successes.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                barrier.wait();
                drop(result);
            });
        }
    });
    assert_eq!(successes.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(Arc::strong_count(&original), isize::MAX as usize - slow_batch);
    drop(reserved);
    assert_eq!(Arc::strong_count(&original), 1);
}

#[test]
fn uninhabited() {
    enum Void {}
    let mut a = Weak::<Void>::new();
    a = a.clone();
    assert!(a.upgrade().is_none());

    let mut a: Weak<dyn Any> = a; // Unsizing
    a = a.clone();
    assert!(a.upgrade().is_none());
}

#[test]
fn slice() {
    let a: Arc<[u32; 3]> = Arc::new([3, 2, 1]);
    let a: Arc<[u32]> = a; // Unsizing
    let b: Arc<[u32]> = Arc::from(&[3, 2, 1][..]); // Conversion
    assert_eq!(a, b);

    // Exercise is_dangling() with a DST
    let mut a = Arc::downgrade(&a);
    a = a.clone();
    assert!(a.upgrade().is_some());
}

#[test]
fn trait_object() {
    let a: Arc<u32> = Arc::new(4);
    let a: Arc<dyn Any> = a; // Unsizing

    // Exercise is_dangling() with a DST
    let mut a = Arc::downgrade(&a);
    a = a.clone();
    assert!(a.upgrade().is_some());

    let mut b = Weak::<u32>::new();
    b = b.clone();
    assert!(b.upgrade().is_none());
    let mut b: Weak<dyn Any> = b; // Unsizing
    b = b.clone();
    assert!(b.upgrade().is_none());
}

#[test]
fn float_nan_ne() {
    let x = Arc::new(f32::NAN);
    assert!(x != x);
    assert!(!(x == x));
}

#[test]
fn partial_eq() {
    struct TestPEq(RefCell<usize>);
    impl PartialEq for TestPEq {
        fn eq(&self, other: &TestPEq) -> bool {
            *self.0.borrow_mut() += 1;
            *other.0.borrow_mut() += 1;
            true
        }
    }
    let x = Arc::new(TestPEq(RefCell::new(0)));
    assert!(x == x);
    assert!(!(x != x));
    assert_eq!(*x.0.borrow(), 4);
}

#[test]
fn eq() {
    #[derive(Eq)]
    struct TestEq(RefCell<usize>);
    impl PartialEq for TestEq {
        fn eq(&self, other: &TestEq) -> bool {
            *self.0.borrow_mut() += 1;
            *other.0.borrow_mut() += 1;
            true
        }
    }
    let x = Arc::new(TestEq(RefCell::new(0)));
    assert!(x == x);
    assert!(!(x != x));
    assert_eq!(*x.0.borrow(), 0);
}

#[test]
fn eq_unsized() {
    #[derive(Eq)]
    struct TestEq<T: ?Sized>(RefCell<usize>, T);
    impl<T: ?Sized> PartialEq for TestEq<T> {
        fn eq(&self, other: &TestEq<T>) -> bool {
            *self.0.borrow_mut() += 1;
            *other.0.borrow_mut() += 1;
            true
        }
    }
    let x = Arc::<TestEq<[u8; 3]>>::new(TestEq(RefCell::new(0), [0, 1, 2])) as Arc<TestEq<[u8]>>;
    assert!(x == x);
    assert!(!(x != x));
    assert_eq!(*x.0.borrow(), 0);
}

#[test]
fn eq_unsized_slice() {
    let a: Arc<[()]> = Arc::new([(); 3]);
    let ptr: *const () = Arc::into_raw(a.clone()).cast();
    let b: Arc<[()]> = unsafe { Arc::from_raw(ptr.cast_slice(42)) };
    assert!(a == a);
    assert!(!(a != a));
    assert!(a != b);
    assert!(!(a == b));
}

// The test code below is identical to that in `rc.rs`.
// For better maintainability we therefore define this type alias.
type Rc<T, A = std::alloc::Global> = Arc<T, A>;

const SHARED_ITER_MAX: u16 = 100;

fn assert_trusted_len<I: TrustedLen>(_: &I) {}

#[test]
fn shared_from_iter_normal() {
    // Exercise the base implementation for non-`TrustedLen` iterators.
    {
        // `Filter` is never `TrustedLen` since we don't
        // know statically how many elements will be kept:
        let iter = (0..SHARED_ITER_MAX).filter(|x| x % 2 == 0).map(Box::new);

        // Collecting into a `Vec<T>` or `Rc<[T]>` should make no difference:
        let vec = iter.clone().collect::<Vec<_>>();
        let rc = iter.collect::<Rc<[_]>>();
        assert_eq!(&*vec, &*rc);

        // Clone a bit and let these get dropped.
        {
            let _rc_2 = rc.clone();
            let _rc_3 = rc.clone();
            let _rc_4 = Rc::downgrade(&_rc_3);
        }
    } // Drop what hasn't been here.
}

#[test]
fn shared_from_iter_trustedlen_normal() {
    // Exercise the `TrustedLen` implementation under normal circumstances
    // where `size_hint()` matches `(_, Some(exact_len))`.
    {
        let iter = (0..SHARED_ITER_MAX).map(Box::new);
        assert_trusted_len(&iter);

        // Collecting into a `Vec<T>` or `Rc<[T]>` should make no difference:
        let vec = iter.clone().collect::<Vec<_>>();
        let rc = iter.collect::<Rc<[_]>>();
        assert_eq!(&*vec, &*rc);
        assert_eq!(size_of::<Box<u16>>() * SHARED_ITER_MAX as usize, size_of_val(&*rc));

        // Clone a bit and let these get dropped.
        {
            let _rc_2 = rc.clone();
            let _rc_3 = rc.clone();
            let _rc_4 = Rc::downgrade(&_rc_3);
        }
    } // Drop what hasn't been here.

    // Try a ZST to make sure it is handled well.
    {
        let iter = (0..SHARED_ITER_MAX).map(drop);
        let vec = iter.clone().collect::<Vec<_>>();
        let rc = iter.collect::<Rc<[_]>>();
        assert_eq!(&*vec, &*rc);
        assert_eq!(0, size_of_val(&*rc));
        {
            let _rc_2 = rc.clone();
            let _rc_3 = rc.clone();
            let _rc_4 = Rc::downgrade(&_rc_3);
        }
    }
}

#[test]
#[should_panic = "I've almost got 99 problems."]
fn shared_from_iter_trustedlen_panic() {
    // Exercise the `TrustedLen` implementation when `size_hint()` matches
    // `(_, Some(exact_len))` but where `.next()` drops before the last iteration.
    let iter = (0..SHARED_ITER_MAX).map(|val| match val {
        98 => panic!("I've almost got 99 problems."),
        _ => Box::new(val),
    });
    assert_trusted_len(&iter);
    let _ = iter.collect::<Rc<[_]>>();

    panic!("I am unreachable.");
}

#[test]
fn shared_from_iter_trustedlen_no_fuse() {
    // Exercise the `TrustedLen` implementation when `size_hint()` matches
    // `(_, Some(exact_len))` but where the iterator does not behave in a fused manner.
    struct Iter(std::vec::IntoIter<Option<Box<u8>>>);

    unsafe impl TrustedLen for Iter {}

    impl Iterator for Iter {
        fn size_hint(&self) -> (usize, Option<usize>) {
            (2, Some(2))
        }

        type Item = Box<u8>;

        fn next(&mut self) -> Option<Self::Item> {
            self.0.next().flatten()
        }
    }

    let vec = vec![Some(Box::new(42)), Some(Box::new(24)), None, Some(Box::new(12))];
    let iter = Iter(vec.into_iter());
    assert_trusted_len(&iter);
    assert_eq!(&[Box::new(42), Box::new(24)], &*iter.collect::<Rc<[_]>>());
}

#[test]
fn weak_may_dangle() {
    fn hmm<'a>(val: &'a mut Weak<&'a str>) -> Weak<&'a str> {
        val.clone()
    }

    // Without #[may_dangle] we get:
    let mut val = Weak::new();
    hmm(&mut val);
    //  ~~~~~~~~ borrowed value does not live long enough
    //
    // `val` dropped here while still borrowed
    // borrow might be used here, when `val` is dropped and runs the `Drop` code for type `std::sync::Weak`
}

/// Test that a panic from a destructor does not leak the allocation.
#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn panic_no_leak() {
    use std::alloc::{AllocError, Allocator, Global, Layout};
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::ptr::NonNull;

    struct AllocCount(Cell<i32>);
    unsafe impl Allocator for AllocCount {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            self.0.set(self.0.get() + 1);
            Global.allocate(layout)
        }
        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            self.0.set(self.0.get() - 1);
            unsafe { Global.deallocate(ptr, layout) }
        }
    }

    struct PanicOnDrop;
    impl Drop for PanicOnDrop {
        fn drop(&mut self) {
            panic!("PanicOnDrop");
        }
    }

    let alloc = AllocCount(Cell::new(0));
    let rc = Rc::new_in(PanicOnDrop, &alloc);
    assert_eq!(alloc.0.get(), 1);

    let panic_message = catch_unwind(AssertUnwindSafe(|| drop(rc))).unwrap_err();
    assert_eq!(*panic_message.downcast_ref::<&'static str>().unwrap(), "PanicOnDrop");
    assert_eq!(alloc.0.get(), 0);
}

/// This is similar to the doc-test for `Arc::make_mut()`, but on an unsized type (slice).
#[test]
fn make_mut_unsized() {
    use alloc::sync::Arc;

    let mut data: Arc<[i32]> = Arc::new([10, 20, 30]);

    Arc::make_mut(&mut data)[0] += 1; // Won't clone anything
    let mut other_data = Arc::clone(&data); // Won't clone inner data
    Arc::make_mut(&mut data)[1] += 1; // Clones inner data
    Arc::make_mut(&mut data)[2] += 1; // Won't clone anything
    Arc::make_mut(&mut other_data)[0] *= 10; // Won't clone anything

    // Now `data` and `other_data` point to different allocations.
    assert_eq!(*data, [11, 21, 31]);
    assert_eq!(*other_data, [110, 20, 30]);
}

#[test]
fn test_unique_arc_weak() {
    let data = UniqueArc::new(32);

    // Test that `Weak` downgraded from `UniqueArc` cannot be upgraded.
    let weak = UniqueArc::downgrade(&data);
    assert_eq!(weak.strong_count(), 0);
    assert_eq!(weak.weak_count(), 0);
    assert!(weak.upgrade().is_none());

    // Test that `Weak` can now be upgraded after the `UniqueArc` being converted to `Arc`.
    let strong = UniqueArc::into_arc(data);
    assert_eq!(*strong, 32);
    assert_eq!(weak.strong_count(), 1);
    assert_eq!(weak.weak_count(), 1);
    let upgraded = weak.upgrade().unwrap();
    assert_eq!(*upgraded, 32);
    assert_eq!(weak.strong_count(), 2);
    assert_eq!(weak.weak_count(), 1);
}

#[allow(unused)]
mod pin_coerce_unsized {
    use alloc::sync::{Arc, UniqueArc};
    use core::pin::Pin;

    pub trait MyTrait {}
    impl MyTrait for String {}

    // Pin coercion should work for Arc
    pub fn pin_arc(arg: Pin<Arc<String>>) -> Pin<Arc<dyn MyTrait>> {
        arg
    }
    pub fn pin_unique_arc(arg: Pin<UniqueArc<String>>) -> Pin<UniqueArc<dyn MyTrait>> {
        arg
    }
}

/// Test that `Arc::make_mut` does not forget an allocator when it steals the data.
#[test]
fn issue_158875_make_mut_dont_leak_allocator() {
    use std::alloc::Global;

    let alloc = Rc::new(Global);

    {
        let mut arc = Arc::new_in(123, alloc.clone());
        let weak = Arc::downgrade(&arc); // create a weak so make_mut steals the data
        _ = Arc::make_mut(&mut arc);
        assert_eq!(weak.upgrade(), None);
    }

    assert_eq!(Rc::strong_count(&alloc), 1); // if this is >1, we have a memory leak!
}

#[test]
#[should_panic = "capacity overflow"]
fn new_uninit_slice_capacity_overflow() {
    let _ = Arc::<[u8]>::new_uninit_slice(isize::MAX as usize);
}

mod arc_allocator_provenance {
    //! Regression tests for issues where the pointer passed back to the allocator
    //! only had the provenance of a reborrow, which is unsound with allocators
    //! that store metadata next to the allocation.
    //! Mainly meant to be run in Miri (but should also work outside it).

    use std::alloc::{AllocError, Allocator, Global, Layout};
    use std::ptr::NonNull;
    use std::sync::Arc;

    struct MyMetadataAlloc;

    fn widen(layout: Layout) -> Layout {
        Layout::from_size_align(layout.size() + 10, layout.align())
            .unwrap_or_else(|_| std::process::abort())
    }

    unsafe impl Allocator for MyMetadataAlloc {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            // imagine we are storing metadata in the extra bytes
            let ptr = Global.allocate(widen(layout))?;
            Ok(NonNull::slice_from_raw_parts(ptr.cast(), layout.size()))
        }
        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            // SAFETY: we get back a pointer with the same provenance as was
            // returned by `allocate`, so it's okay to deallocate the full
            // `layout.size() + 10` bytes through it.
            unsafe { Global.deallocate(ptr, widen(layout)) }
        }
    }

    #[test]
    fn issue_162719() {
        drop(Arc::<i32, _>::new_uninit_in(MyMetadataAlloc));
        drop(Arc::new_in(1i32, MyMetadataAlloc));
    }

    #[test]
    fn issue_162720() {
        drop(Arc::new_cyclic_in(|_| 1i32, MyMetadataAlloc));
    }
}
