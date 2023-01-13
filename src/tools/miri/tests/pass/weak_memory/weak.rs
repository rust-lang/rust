//@compile-flags: -Zmiri-ignore-leaks -Zmiri-preemption-rate=0

// Tests showing weak memory behaviours are exhibited. All tests
// return true when the desired behaviour is seen.
// This is scheduler and pseudo-RNG dependent, so each test is
// run multiple times until one try returns true.
// Spurious failure is possible, if you are really unlucky with
// the RNG and always read the latest value from the store buffer.

use std::sync::atomic::Ordering::*;
use std::sync::atomic::{fence, AtomicUsize};
use std::thread::spawn;

#[derive(Copy, Clone)]
struct EvilSend<T>(pub T);

unsafe impl<T> Send for EvilSend<T> {}
unsafe impl<T> Sync for EvilSend<T> {}

// We can't create static items because we need to run each test
// multiple times
fn static_atomic(val: usize) -> &'static AtomicUsize {
    let ret = Box::leak(Box::new(AtomicUsize::new(val)));
    ret
}

// Spins until it reads the given value
fn reads_value(loc: &AtomicUsize, val: usize) -> usize {
    while loc.load(Relaxed) != val {
        std::hint::spin_loop();
    }
    val
}

fn relaxed() -> bool {
    let x = static_atomic(0);
    let j1 = spawn(move || {
        x.store(1, Relaxed);
        x.store(2, Relaxed);
    });

    let j2 = spawn(move || x.load(Relaxed));

    j1.join().unwrap();
    let r2 = j2.join().unwrap();

    r2 == 1
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
    assert_eq!(x.load(Relaxed), 11); // work around https://github.com/rust-lang/miri/issues/2164

    let wait = static_atomic(0);

    let j1 = spawn(move || {
        x.store(22, Relaxed);
        // Relaxed is intentional. We want to test if the thread 2 reads the initialisation write
        // after a relaxed write
        wait.store(1, Relaxed);
    });

    let j2 = spawn(move || {
        reads_value(wait, 1);
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
    assert_eq!(x.load(Relaxed), 0); // work around https://github.com/rust-lang/miri/issues/2164
    let y = static_atomic(0);
    assert_eq!(y.load(Relaxed), 0); // work around https://github.com/rust-lang/miri/issues/2164
    let z = static_atomic(0);
    assert_eq!(z.load(Relaxed), 0); // work around https://github.com/rust-lang/miri/issues/2164

    // Since each thread is so short, we need to make sure that they truely run at the same time
    // Otherwise t1 will finish before t2 even starts
    let go = static_atomic(0);

    let t1 = spawn(move || {
        while go.load(Relaxed) == 0 {}
        rdmw(y, x, z)
    });

    let t2 = spawn(move || {
        while go.load(Relaxed) == 0 {}
        rdmw(z, x, y)
    });

    go.store(1, Relaxed);

    let a = t1.join().unwrap();
    let b = t2.join().unwrap();
    (a, b) == (0, 0)
}

/// Asserts that the function returns true at least once in 100 runs
fn assert_once(f: fn() -> bool) {
    assert!(std::iter::repeat_with(|| f()).take(100).any(|x| x));
}

pub fn main() {
    assert_once(relaxed);
    assert_once(seq_cst);
    assert_once(|| initialization_write(false));
    assert_once(|| initialization_write(true));
    assert_once(faa_replaced_by_load);
}
