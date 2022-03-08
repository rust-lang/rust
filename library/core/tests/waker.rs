use std::ptr;
use std::task::{RawWaker, RawWakerVTable, Waker};

#[test]
fn test_waker_getters() {
    let raw_waker = RawWaker::new(42usize as *mut (), &WAKER_VTABLE);
    assert_eq!(raw_waker.data() as usize, 42);
    assert!(ptr::eq(raw_waker.vtable(), &WAKER_VTABLE));

    let waker = unsafe { Waker::from_raw(raw_waker) };
    let waker2 = waker.clone();
    let raw_waker2 = waker2.as_raw();
    assert_eq!(raw_waker2.data() as usize, 43);
    assert!(ptr::eq(raw_waker2.vtable(), &WAKER_VTABLE));
}

static WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(
    |data| RawWaker::new((data as usize + 1) as *mut (), &WAKER_VTABLE),
    |_| {},
    |_| {},
    |_| {},
);
