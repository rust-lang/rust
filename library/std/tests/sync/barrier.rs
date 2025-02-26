use std::sync::mpsc::{TryRecvError, channel};
use std::sync::{Arc, Barrier};
use std::thread;

#[test]
#[cfg_attr(any(target_os = "emscripten", target_os = "wasi"), ignore)] // no threads
fn test_barrier() {
    const N: usize = 10;

    let barrier = Arc::new(Barrier::new(N));
    let (tx, rx) = channel();

    for _ in 0..N - 1 {
        let c = barrier.clone();
        let tx = tx.clone();
        thread::spawn(move || {
            tx.send(c.wait().is_leader()).unwrap();
        });
    }

    // At this point, all spawned threads should be blocked,
    // so we shouldn't get anything from the port
    assert!(matches!(rx.try_recv(), Err(TryRecvError::Empty)));

    let mut leader_found = barrier.wait().is_leader();

    // Now, the barrier is cleared and we should get data.
    for _ in 0..N - 1 {
        if rx.recv().unwrap() {
            assert!(!leader_found);
            leader_found = true;
        }
    }
    assert!(leader_found);
}
