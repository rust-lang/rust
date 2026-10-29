// This specifically tests behavior *without* preemption.
//@compile-flags: -Zmiri-deterministic-concurrency

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;

/// When a thread yields, Miri's scheduler used to pick the thread with the lowest ID
/// that can run. IDs are assigned in thread creation order.
/// This means we could make 2 threads infinitely ping-pong with each other while
/// really there is a 3rd thread that we should schedule to make progress.
fn two_player_ping_pong() {
    static FLAG: AtomicUsize = AtomicUsize::new(0);

    let waiter1 = thread::spawn(|| {
        while FLAG.load(Ordering::Acquire) == 0 {
            // spin and wait
            thread::yield_now();
        }
    });
    let waiter2 = thread::spawn(|| {
        while FLAG.load(Ordering::Acquire) == 0 {
            // spin and wait
            thread::yield_now();
        }
    });
    let progress = thread::spawn(|| {
        FLAG.store(1, Ordering::Release);
    });
    // The first `join` blocks the main thread and thus takes it out of the equation.
    waiter1.join().unwrap();
    waiter2.join().unwrap();
    progress.join().unwrap();
}

/// Based on a test by @jethrogb.
fn launcher() {
    static THREAD2_LAUNCHED: AtomicBool = AtomicBool::new(false);

    for _ in 0..10 {
        let (tx, rx) = mpsc::sync_channel(0);
        THREAD2_LAUNCHED.store(false, Ordering::SeqCst);

        let jh = thread::spawn(move || {
            struct RecvOnDrop(Cell<Option<mpsc::Receiver<()>>>);

            impl Drop for RecvOnDrop {
                fn drop(&mut self) {
                    let rx = self.0.take().unwrap();
                    while !THREAD2_LAUNCHED.load(Ordering::SeqCst) {
                        thread::yield_now();
                    }
                    rx.recv().unwrap();
                }
            }

            let tl_rx: RecvOnDrop = RecvOnDrop(Cell::new(None));
            tl_rx.0.set(Some(rx));
        });

        let tx_clone = tx.clone();
        let jh2 = thread::spawn(move || {
            THREAD2_LAUNCHED.store(true, Ordering::SeqCst);
            jh.join().unwrap();
            tx_clone.send(()).expect_err(
                "Expecting channel to be closed because thread 1 TLS destructors must've run",
            );
        });

        while !THREAD2_LAUNCHED.load(Ordering::SeqCst) {
            thread::yield_now();
        }
        thread::yield_now();
        tx.send(()).expect("Expecting channel to be live because thread 2 must block on join");
        jh2.join().unwrap();
    }
}

fn main() {
    two_player_ping_pong();
    launcher();
}
