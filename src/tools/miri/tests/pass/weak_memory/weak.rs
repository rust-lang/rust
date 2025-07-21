//@compile-flags: -Zmiri-ignore-leaks -Zmiri-fixed-schedule

// Tests showing weak memory behaviours are exhibited. All tests
// return true when the desired behaviour is seen.
// This is scheduler and pseudo-RNG dependent, so each test is
// run multiple times until one try returns true.
// Spurious failure is possible, if you are really unlucky with
// the RNG and always read the latest value from the store buffer.

use std::sync::atomic::Ordering::*;
use std::sync::atomic::{AtomicUsize, fence};
use std::thread::spawn;

#[allow(dead_code)]
#[derive(Copy, Clone)]
struct EvilSend<T>(pub T);

unsafe impl<T> Send for EvilSend<T> {}
unsafe impl<T> Sync for EvilSend<T> {}

// We can't create static items because we need to run each test multiple times.
fn static_atomic(val: usize) -> &'static AtomicUsize {
    Box::leak(Box::new(AtomicUsize::new(val)))
}

// Spins until it reads the given value
fn spin_until(loc: &AtomicUsize, val: usize) -> usize {
    while loc.load(Relaxed) != val {
        std::hint::spin_loop();
    }
    val
}

fn relaxed(initial_read: bool) -> bool {
    let x = static_atomic(0);
    let j1 = spawn(move || {
        x.store(1, Relaxed);
        // Preemption is disabled, so the store above will never be the
        // latest store visible to another thread.
        x.store(2, Relaxed);
    });

    let j2 = spawn(move || x.load(Relaxed));

    j1.join().unwrap();
    let r2 = j2.join().unwrap();

    // There are three possible values here: 0 (from the initial read), 1 (from the first relaxed
    // read), and 2 (the last read). The last case is boring and we cover the other two.
    r2 == if initial_read { 0 } else { 1 }
}

// https://www.doc.ic.ac.uk/~afd/homepages/papers/pdfs/2017/POPL.pdf Figure 8
fn seq_cst() -> bool {
    let x = static_atomic(0);

    let j1 = spawn(move || {
        x.store(1, Relaxed);
    });

    let j2 = spawn(move || {
        x.store(2, SeqCst);
        x.store(3, SeqCst);
    });

    let j3 = spawn(move || x.load(SeqCst));

    j1.join().unwrap();
    j2.join().unwrap();
    let r3 = j3.join().unwrap();

    r3 == 1
}

fn initialization_write(add_fence: bool) -> bool {
    let x = static_atomic(11);

    let wait = static_atomic(0);

    let j1 = spawn(move || {
        x.store(22, Relaxed);
        // Relaxed is intentional. We want to test if the thread 2 reads the initialisation write
        // after a relaxed write
        wait.store(1, Relaxed);
    });

    let j2 = spawn(move || {
        spin_until(wait, 1);
        if add_fence {
            fence(AcqRel);
        }
        x.load(Relaxed)
    });

    j1.join().unwrap();
    let r2 = j2.join().unwrap();

    r2 == 11
}

fn faa_replaced_by_load() -> bool {
    // Example from https://github.com/llvm/llvm-project/issues/56450#issuecomment-1183695905
    #[no_mangle]
    pub fn rdmw(storing: &AtomicUsize, sync: &AtomicUsize, loading: &AtomicUsize) -> usize {
        storing.store(1, Relaxed);
        fence(Release);
        // sync.fetch_add(0, Relaxed);
        sync.load(Relaxed);
        fence(Acquire);
        loading.load(Relaxed)
    }

    let x = static_atomic(0);
    let y = static_atomic(0);
    let z = static_atomic(0);

    // Since each thread is so short, we need to make sure that they truely run at the same time
    // Otherwise t1 will finish before t2 even starts
    let go = static_atomic(0);

    let t1 = spawn(move || {
        spin_until(go, 1);
        rdmw(y, x, z)
    });

    let t2 = spawn(move || {
        spin_until(go, 1);
        rdmw(z, x, y)
    });

    go.store(1, Relaxed);

    let a = t1.join().unwrap();
    let b = t2.join().unwrap();
    (a, b) == (0, 0)
}

/// Asserts that the function returns true at least once in 100 runs
#[track_caller]
fn assert_once(f: fn() -> bool) {
    assert!(std::iter::repeat_with(|| f()).take(100).any(|x| x));
}

pub fn main() {
    assert_once(|| relaxed(false));
    assert_once(|| relaxed(true));
    assert_once(seq_cst);
    assert_once(|| initialization_write(false));
    assert_once(|| initialization_write(true));
    assert_once(faa_replaced_by_load);
}
