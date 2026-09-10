use super::*;
use crate::sync::Arc;
use crate::thread;

#[test]
fn queue() {
    let wq = Arc::new(WaitVariable::new(()));
    let wq2 = wq.clone();

    let locked = (*wq).as_ref().lock_pinned();

    let t1 = thread::spawn(move || {
        // if we obtain the lock, the main thread should be waiting
        assert!(WaitQueue::notify_one((*wq2).as_ref().lock_pinned()).is_ok());
    });

    WaitQueue::wait(locked, || {});

    t1.join().unwrap();
}
