use std::ops::Deref;

use super::OwnedSlice;

#[test]
fn create_deref() {
    let owned_slice = OwnedSlice::new(Box::new(vec![1, 2, 3]));

    let slice = &*owned_slice;

    assert_eq!(slice, &[1, 2, 3]);
}

#[test]
fn map() {
    let owned_slice: OwnedSlice<Box<dyn Deref<Target = [u8]>>, _> =
        OwnedSlice::new(Box::new(vec![1, 2, 3, 4, 5]));

    let owned_slice = owned_slice.map(|slice| slice.split_at(2).1);
    let slice = &*owned_slice;

    assert_eq!(slice, &[3, 4, 5]);
}

#[test]
fn empty_slice() {
    let owned_slice = OwnedSlice::new(Box::new(vec![1, 2, 3, 4, 5]));

    let owned_slice = owned_slice.map(|slice| &slice[0..0]);

    let slice = &*owned_slice;

    assert_eq!(slice, &[]);
}

#[test]
#[should_panic]
fn out_of_bounds() {
    static X: [u8; 5] = [1, 2, 3, 4, 5];

    let owned_slice = OwnedSlice::new(Box::new(vec![1u8, 2, 3]));
    let owned_slice = owned_slice.map(|_| &X[..]);
    let slice = &*owned_slice;

    assert_eq!(slice, &[1, 2, 3, 4, 5]);
}

#[test]
#[should_panic]
fn no_zsts_allowed() {
    let other = Box::leak(Box::new(vec![(); 5]));
    ignore_leak(other);

    let owned_slice = OwnedSlice::new(Box::new(vec![(); 5]));
    let owned_slice = owned_slice.map(|_| &other[..]);
    let slice = &*owned_slice;

    assert_eq!(slice, other);
}

/// It's ok for this to leak, we need a 'static reference.
fn ignore_leak<T>(_ptr: *const T) {
    #[cfg(miri)]
    extern "Rust" {
        fn miri_static_root(ptr: *const u8);
    }
    #[cfg(miri)]
    unsafe {
        miri_static_root(_ptr.cast())
    };
}
