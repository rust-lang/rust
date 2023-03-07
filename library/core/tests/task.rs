use core::task::{Poll, RawWaker, RawWakerVTable, Waker};

#[test]
fn poll_const() {
    // test that the methods of `Poll` are usable in a const context

    const POLL: Poll<usize> = Poll::Pending;

    const IS_READY: bool = POLL.is_ready();
    assert!(!IS_READY);

    const IS_PENDING: bool = POLL.is_pending();
    assert!(IS_PENDING);
}

#[test]
fn waker_const() {
    const VOID_TABLE: RawWakerVTable = RawWakerVTable::new(|_| VOID_WAKER, |_| {}, |_| {}, |_| {});

    const VOID_WAKER: RawWaker = RawWaker::new(&(), &VOID_TABLE);

    static WAKER: Waker = unsafe { Waker::from_raw(VOID_WAKER) };

    WAKER.wake_by_ref();
}
