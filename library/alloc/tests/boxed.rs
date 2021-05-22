use std::alloc::{self, AllocError, Allocator, Layout};
use std::any::Any;
use std::cell::Cell;
use std::mem::MaybeUninit;
use std::panic;
use std::ptr::NonNull;

#[test]
fn unitialized_zero_size_box() {
    assert_eq!(
        &*Box::<()>::new_uninit() as *const _,
        NonNull::<MaybeUninit<()>>::dangling().as_ptr(),
    );
    assert_eq!(
        Box::<[()]>::new_uninit_slice(4).as_ptr(),
        NonNull::<MaybeUninit<()>>::dangling().as_ptr(),
    );
    assert_eq!(
        Box::<[String]>::new_uninit_slice(0).as_ptr(),
        NonNull::<MaybeUninit<String>>::dangling().as_ptr(),
    );
}

#[test]
fn box_reuse_slice() {
    let mut numbers: Box<[i32]> = Box::new([1, 2, 3]);

    Box::reuse(&mut numbers, [4, 5, 6]).unwrap();
    assert_eq!(&*numbers, &[4, 5, 6]);

    Box::reuse(&mut numbers, [0, 0]).unwrap_err();
    assert_eq!(&*numbers, &[4, 5, 6]);

    Box::reuse(&mut numbers, [0, 0, 0, 0]).unwrap_err();
    assert_eq!(&*numbers, &[4, 5, 6]);
}

#[test]
fn box_reuse_dyn() {
    let mut boxed: Box<dyn ToString> = Box::new(5_usize);

    Box::reuse(&mut boxed, 9_usize).unwrap();
    assert_eq!(boxed.to_string(), "9");

    Box::reuse(&mut boxed, 6_i32).unwrap_err();

    Box::reuse(&mut boxed, 26_isize).unwrap();
    assert_eq!(boxed.to_string(), "26");
}

#[test]
fn box_reuse_panic_drop() {
    struct PanicsOnDrop;
    impl Drop for PanicsOnDrop {
        fn drop(&mut self) {
            panic::panic_any(15_i32);
        }
    }

    let mut boxed: Box<dyn Any> = Box::new(PanicsOnDrop);

    let payload =
        panic::catch_unwind(panic::AssertUnwindSafe(|| Box::reuse(&mut boxed, ()))).unwrap_err();
    assert_eq!(*payload.downcast::<i32>().unwrap(), 15);

    assert!(boxed.downcast_ref::<PanicsOnDrop>().is_none());
    boxed.downcast_ref::<()>().unwrap();

    assert!(Box::reuse(&mut boxed, PanicsOnDrop).is_ok());

    let payload =
        panic::catch_unwind(panic::AssertUnwindSafe(|| Box::set(&mut boxed, 2))).unwrap_err();
    assert_eq!(*payload.downcast::<i32>().unwrap(), 15);

    assert!(boxed.downcast_ref::<PanicsOnDrop>().is_none());
    assert!(boxed.downcast_ref::<()>().is_none());
    assert_eq!(*boxed.downcast_ref::<i32>().unwrap(), 2);
}

#[test]
fn box_set_allocator() {
    // `Box::set` does some tricky things to create a new box with the same allocator.

    thread_local! {
        static ALLOCATIONS: Cell<usize> = Cell::new(0);
        static DROPPED: Cell<bool> = Cell::new(false);
    }

    struct Alloc;
    unsafe impl Allocator for Alloc {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            ALLOCATIONS.with(|allocations| allocations.set(allocations.get() + 1));
            alloc::Global.allocate(layout)
        }
        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            alloc::Global.deallocate(ptr, layout)
        }
    }
    impl Drop for Alloc {
        fn drop(&mut self) {
            let dropped = DROPPED.with(|dropped| dropped.replace(true));
            assert!(!dropped, "Box::set dropped allocator twice");
        }
    }

    ALLOCATIONS.with(|allocations| allocations.set(0));
    DROPPED.with(|dropped| dropped.set(false));

    let mut boxed: Box<[i32], Alloc> = Box::new_in([1, 2, 3], Alloc);
    assert_eq!(ALLOCATIONS.with(Cell::get), 1);

    Box::set(&mut boxed, [4, 5, 6]);
    assert_eq!(ALLOCATIONS.with(Cell::get), 1);

    Box::set(&mut boxed, [1, 2, 3, 4]);
    assert_eq!(ALLOCATIONS.with(Cell::get), 2);

    Box::set(&mut boxed, []);
    assert_eq!(ALLOCATIONS.with(Cell::get), 3);

    drop(boxed);
    assert!(DROPPED.with(Cell::get));
}

#[derive(Clone, PartialEq, Eq, Debug)]
struct Dummy {
    _data: u8,
}

#[test]
fn box_clone_and_clone_from_equivalence() {
    for size in (0..8).map(|i| 2usize.pow(i)) {
        let control = vec![Dummy { _data: 42 }; size].into_boxed_slice();
        let clone = control.clone();
        let mut copy = vec![Dummy { _data: 84 }; size].into_boxed_slice();
        copy.clone_from(&control);
        assert_eq!(control, clone);
        assert_eq!(control, copy);
    }
}

/// This test might give a false positive in case the box realocates, but the alocator keeps the
/// original pointer.
///
/// On the other hand it won't give a false negative, if it fails than the memory was definitely not
/// reused
#[test]
fn box_clone_from_ptr_stability() {
    for size in (0..8).map(|i| 2usize.pow(i)) {
        let control = vec![Dummy { _data: 42 }; size].into_boxed_slice();
        let mut copy = vec![Dummy { _data: 84 }; size].into_boxed_slice();
        let copy_raw = copy.as_ptr() as usize;
        copy.clone_from(&control);
        assert_eq!(copy.as_ptr() as usize, copy_raw);
    }
}

#[test]
fn box_deref_lval() {
    let x = Box::new(Cell::new(5));
    x.set(1000);
    assert_eq!(x.get(), 1000);
}
