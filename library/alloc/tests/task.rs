use alloc::rc::Rc;
use alloc::sync::Arc;
use alloc::task::{LocalWake, Wake};
use core::task::{LocalWaker, Waker};

#[test]
fn test_waker_will_wake_clone() {
    struct NoopWaker;

    impl Wake for NoopWaker {
        fn wake(self: Arc<Self>) {}
    }

    let waker = Waker::from(Arc::new(NoopWaker));
    let clone = waker.clone();

    assert!(waker.will_wake(&clone));
    assert!(clone.will_wake(&waker));
}

#[test]
fn test_local_waker_will_wake_clone() {
    struct NoopWaker;

    impl LocalWake for NoopWaker {
        fn wake(self: Rc<Self>) {}
    }

    let waker = LocalWaker::from(Rc::new(NoopWaker));
    let clone = waker.clone();

    assert!(waker.will_wake(&clone));
    assert!(clone.will_wake(&waker));
}
