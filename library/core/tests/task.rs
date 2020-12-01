use core::task::Poll;

#[test]
fn poll_const() {
    // test that the methods of `Poll` are usable in a const context

    const POLL: Poll<usize> = Poll::Pending;

    const IS_READY: bool = POLL.is_ready();
    assert!(!IS_READY);

    const IS_PENDING: bool = POLL.is_pending();
    assert!(IS_PENDING);
}
