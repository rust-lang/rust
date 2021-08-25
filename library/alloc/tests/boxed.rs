use std::cell::Cell;
use std::mem::MaybeUninit;
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

/// This test might give a false positive in case the box reallocates,
/// but the allocator keeps the original pointer.
///
/// On the other hand, it won't give a false negative: If it fails, then the
/// memory was definitely not reused.
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
