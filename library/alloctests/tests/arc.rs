use std::any::Any;
use std::cell::{Cell, RefCell};
use std::iter::TrustedLen;
use std::sync::{Arc, UniqueArc, Weak};

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
