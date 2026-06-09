use super::*;
use crate::sync::Arc;
use crate::thread;

#[test]
fn queue() {
    let wq = Arc::new(SpinMutex::<WaitVariable<()>>::default());
    let wq2 = wq.clone();

    let locked = wq.lock();

    let t1 = thread::spawn(move || {
        // if we obtain the lock, the main thread should be waiting
        assert!(WaitQueue::notify_one(wq2.lock()).is_ok());
    });

    WaitQueue::wait(locked, || {});

    t1.join().unwrap();
}
