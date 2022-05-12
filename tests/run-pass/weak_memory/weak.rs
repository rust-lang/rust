// ignore-windows: Concurrency on Windows is not supported yet.
// compile-flags: -Zmiri-ignore-leaks

// Tests showing weak memory behaviours are exhibited. All tests
// return true when the desired behaviour is seen.
// This is scheduler and pseudo-RNG dependent, so each test is
// run multiple times until one try returns true.
// Spurious failure is possible, if you are really unlucky with
// the RNG.

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::*;
use std::thread::spawn;

#[derive(Copy, Clone)]
struct EvilSend<T>(pub T);

unsafe impl<T> Send for EvilSend<T> {}
unsafe impl<T> Sync for EvilSend<T> {}

// We can't create static items because we need to run each test
// multiple times
fn static_atomic(val: usize) -> &'static AtomicUsize {
    let ret = Box::leak(Box::new(AtomicUsize::new(val)));
    // A workaround to put the initialisation value in the store buffer
    ret.store(val, Relaxed);
    ret
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

// Asserts that the function returns true at least once in 100 runs
macro_rules! assert_once {
    ($f:ident) => {
        assert!(std::iter::repeat_with(|| $f()).take(100).any(|x| x));
    };
}

pub fn main() {
    assert_once!(relaxed);
    assert_once!(seq_cst);
}
