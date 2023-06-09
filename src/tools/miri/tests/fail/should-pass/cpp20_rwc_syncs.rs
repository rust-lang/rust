//@compile-flags: -Zmiri-ignore-leaks

// https://plv.mpi-sws.org/scfix/paper.pdf
// 2.2 Second Problem: SC Fences are Too Weak
// This test should pass under the C++20 model Rust is using.
// Unfortunately, Miri's weak memory emulation only follows the C++11 model
// as we don't know how to correctly emulate C++20's revised SC semantics,
// so we have to stick to C++11 emulation from existing research.

use std::sync::atomic::Ordering::*;
use std::sync::atomic::{fence, AtomicUsize};
use std::thread::spawn;

// Spins until it reads the given value
fn reads_value(loc: &AtomicUsize, val: usize) -> usize {
    while loc.load(Relaxed) != val {
        std::hint::spin_loop();
    }
    val
}

// We can't create static items because we need to run each test
// multiple tests
fn static_atomic(val: usize) -> &'static AtomicUsize {
    let ret = Box::leak(Box::new(AtomicUsize::new(val)));
    // A workaround to put the initialization value in the store buffer.
    // See https://github.com/rust-lang/miri/issues/2164
    ret.load(Relaxed);
    ret
}

fn test_cpp20_rwc_syncs() {
    /*
    int main() {
        atomic_int x = 0;
        atomic_int y = 0;

        {{{ x.store(1,mo_relaxed);
        ||| { r1=x.load(mo_relaxed).readsvalue(1);
              fence(mo_seq_cst);
              r2=y.load(mo_relaxed); }
        ||| { y.store(1,mo_relaxed);
              fence(mo_seq_cst);
              r3=x.load(mo_relaxed); }
        }}}
        return 0;
    }
    */
    let x = static_atomic(0);
    let y = static_atomic(0);

    let j1 = spawn(move || {
        x.store(1, Relaxed);
    });

    let j2 = spawn(move || {
        reads_value(&x, 1);
        fence(SeqCst);
        y.load(Relaxed)
    });

    let j3 = spawn(move || {
        y.store(1, Relaxed);
        fence(SeqCst);
        x.load(Relaxed)
    });

    j1.join().unwrap();
    let b = j2.join().unwrap();
    let c = j3.join().unwrap();

    // We cannot write assert_ne!() since ui_test's fail
    // tests expect exit status 1, whereas panics produce 101.
    // Our ui_test does not yet support overriding failure status codes.
    if (b, c) == (0, 0) {
        // This *should* be unreachable, but Miri will reach it.
        unsafe {
            std::hint::unreachable_unchecked(); //~ERROR: unreachable
        }
    }
}

pub fn main() {
    for _ in 0..500 {
        test_cpp20_rwc_syncs();
    }
}
