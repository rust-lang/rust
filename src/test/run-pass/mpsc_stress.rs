// compile-flags:--test
// ignore-emscripten
// ignore-sgx no thread sleep support

use std::sync::mpsc::channel;
use std::sync::mpsc::TryRecvError;
use std::sync::mpsc::RecvError;
use std::sync::mpsc::RecvTimeoutError;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use std::thread;
use std::time::Duration;


/// Simple thread synchronization utility
struct Barrier {
    // Not using mutex/condvar for precision
    shared: Arc<AtomicUsize>,
    count: usize,
}

impl Barrier {
    fn new(count: usize) -> Vec<Barrier> {
        let shared = Arc::new(AtomicUsize::new(0));
        (0..count).map(|_| Barrier { shared: shared.clone(), count: count }).collect()
    }

    fn new2() -> (Barrier, Barrier) {
        let mut v = Barrier::new(2);
        (v.pop().unwrap(), v.pop().unwrap())
    }

    /// Returns when `count` threads enter `wait`
    fn wait(self) {
        self.shared.fetch_add(1, Ordering::SeqCst);
        while self.shared.load(Ordering::SeqCst) != self.count {
        }
    }
}


fn shared_close_sender_does_not_lose_messages_iter() {
    let (tb, rb) = Barrier::new2();

    let (tx, rx) = channel();
    let _ = tx.clone(); // convert to shared

    thread::spawn(move || {
        tb.wait();
        thread::sleep(Duration::from_micros(1));
        tx.send(17).expect("send");
        drop(tx);
    });

    let i = rx.into_iter();
    rb.wait();
    // Make sure it doesn't return disconnected before returning an element
    assert_eq!(vec![17], i.collect::<Vec<_>>());
}

#[test]
fn shared_close_sender_does_not_lose_messages() {
    for _ in 0..10000 {
        shared_close_sender_does_not_lose_messages_iter();
    }
}


// https://github.com/rust-lang/rust/issues/39364
fn concurrent_recv_timeout_and_upgrade_iter() {
    // 1 us
    let sleep = Duration::new(0, 1_000);

    let (a, b) = Barrier::new2();
    let (tx, rx) = channel();
    let th = thread::spawn(move || {
        a.wait();
        loop {
            match rx.recv_timeout(sleep) {
                Ok(_) => {
                    break;
                },
                Err(_) => {},
            }
        }
    });
    b.wait();
    thread::sleep(sleep);
    tx.clone().send(()).expect("send");
    th.join().unwrap();
}

#[test]
fn concurrent_recv_timeout_and_upgrade() {
    // FIXME: fix and enable
    if true { return }

    // at the moment of writing this test fails like this:
    // thread '<unnamed>' panicked at 'assertion failed: `(left == right)`
    //  left: `4561387584`,
    // right: `0`', libstd/sync/mpsc/shared.rs:253:13

    for _ in 0..10000 {
        concurrent_recv_timeout_and_upgrade_iter();
    }
}


fn concurrent_writes_iter() {
    const THREADS: usize = 4;
    const PER_THR: usize = 100;

    let mut bs = Barrier::new(THREADS + 1);
    let (tx, rx) = channel();

    let mut threads = Vec::new();
    for j in 0..THREADS {
        let tx = tx.clone();
        let b = bs.pop().unwrap();
        threads.push(thread::spawn(move || {
            b.wait();
            for i in 0..PER_THR {
                tx.send(j * 1000 + i).expect("send");
            }
        }));
    }

    let b = bs.pop().unwrap();
    b.wait();

    let mut v: Vec<_> = rx.iter().take(THREADS * PER_THR).collect();
    v.sort();

    for j in 0..THREADS {
        for i in 0..PER_THR {
            assert_eq!(j * 1000 + i, v[j * PER_THR + i]);
        }
    }

    for t in threads {
        t.join().unwrap();
    }

    let one_us = Duration::new(0, 1000);

    assert_eq!(TryRecvError::Empty, rx.try_recv().unwrap_err());
    assert_eq!(RecvTimeoutError::Timeout, rx.recv_timeout(one_us).unwrap_err());

    drop(tx);

    assert_eq!(RecvError, rx.recv().unwrap_err());
    assert_eq!(RecvTimeoutError::Disconnected, rx.recv_timeout(one_us).unwrap_err());
    assert_eq!(TryRecvError::Disconnected, rx.try_recv().unwrap_err());
}

#[test]
fn concurrent_writes() {
    for _ in 0..100 {
        concurrent_writes_iter();
    }
}
