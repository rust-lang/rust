// run-pass

use std::task::Poll;

fn main() {
    const POLL : Poll<usize> = Poll::Pending;

    const IS_READY : bool = POLL.is_ready();
    assert!(!IS_READY);

    const IS_PENDING : bool = POLL.is_pending();
    assert!(IS_PENDING);
}
