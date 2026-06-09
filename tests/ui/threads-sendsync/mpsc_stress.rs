//@ run-pass
//@ compile-flags:--test
//@ needs-threads

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, RecvError, RecvTimeoutError, TryRecvError};
use std::sync::Arc;
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
            #[cfg(target_env = "sgx")]
            thread::yield_now();
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
    with_minimum_timer_resolution(|| {
        for _ in 0..10000 {
            shared_close_sender_does_not_lose_messages_iter();
        }
    });
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
                }
                Err(_) => {}
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
    with_minimum_timer_resolution(|| {
        for _ in 0..10000 {
            concurrent_recv_timeout_and_upgrade_iter();
        }
    });
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
    with_minimum_timer_resolution(|| {
        for _ in 0..100 {
            concurrent_writes_iter();
        }
    });
}

#[cfg(windows)]
pub mod timeapi {
    #![allow(non_snake_case)]
    use std::ffi::c_uint;

    pub const TIMERR_NOERROR: c_uint = 0;

    #[link(name = "winmm")]
    extern "system" {
        pub fn timeBeginPeriod(uPeriod: c_uint) -> c_uint;
        pub fn timeEndPeriod(uPeriod: c_uint) -> c_uint;
    }
}

/// Window's minimum sleep time can be as much as 16ms.
// This function evaluates the closure with this resolution
// set as low as possible.
///
/// This takes the above test's duration from 10000*16/1000/60=2.67 minutes to ~16 seconds.
fn with_minimum_timer_resolution(f: impl Fn()) {
    #[cfg(windows)]
    unsafe {
        let ret = timeapi::timeBeginPeriod(1);
        assert_eq!(ret, timeapi::TIMERR_NOERROR);

        f();

        let ret = timeapi::timeEndPeriod(1);
        assert_eq!(ret, timeapi::TIMERR_NOERROR);
    }

    #[cfg(not(windows))]
    {
        f();
    }
}
