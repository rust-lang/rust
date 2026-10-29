//@compile-flags: -Zmiri-ignore-leaks -Zmiri-fixed-schedule
// This test's runtime explodes if the GC interval is set to 1 (which we do in CI), so we
// override it internally back to the default frequency.
//@compile-flags: -Zmiri-provenance-gc=10000

// Tests showing weak memory behaviours are exhibited, even with a fixed scheule.
// We run all tests a number of times and then check that we see the desired list of outcomes.

// Spurious failure is possible, if you are really unlucky with
// the RNG and always read the latest value from the store buffer.

use std::sync::atomic::Ordering::*;
use std::sync::atomic::{AtomicUsize, fence};
use std::thread::spawn;

#[path = "../../utils/mod.rs"]
mod utils;
use utils::check_all_outcomes;

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

fn relaxed() {
    check_all_outcomes([0, 1, 2], || {
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
        // read), and 2 (the last read).
        r2
    });
}

// https://www.doc.ic.ac.uk/~afd/homepages/papers/pdfs/2017/POPL.pdf Figure 8
fn seq_cst() {
    check_all_outcomes([1, 3], || {
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

        // Even though we force t3 to run last, it can still see the value 1.
        // And it can *never* see the value 2!
        r3
    });
}

fn initialization_write(add_fence: bool) {
    check_all_outcomes([11, 22], || {
        let x = static_atomic(11);

        if add_fence {
            // For the fun of it, let's make this location atomic and non-atomic again,
            // to ensure Miri updates the state properly for that.
            x.store(99, Relaxed);
            unsafe { std::ptr::from_ref(x).cast_mut().write(11.into()) };
        }

        let wait = static_atomic(0);

        let j1 = spawn(move || {
            x.store(22, Relaxed);
            // Since nobody else writes to `x`, we can non-atomically read it.
            // (This tests that we do not delete the store buffer here.)
            unsafe { std::ptr::from_ref(x).read() };
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

        r2
    });
}

fn faa_replaced_by_load() {
    check_all_outcomes([true, false], || {
        // Example from https://github.com/llvm/llvm-project/issues/56450#issuecomment-1183695905
        fn rdmw(storing: &AtomicUsize, sync: &AtomicUsize, loading: &AtomicUsize) -> usize {
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

        let t1 = spawn(move || rdmw(y, x, z));

        let t2 = spawn(move || rdmw(z, x, y));

        let a = t1.join().unwrap();
        let b = t2.join().unwrap();
        (a, b) == (0, 0)
    });
}

/// Checking that the weaker release sequence example from
/// <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0982r0.html> can actually produce the
/// new behavior (`Some(0)` in our version).
fn weaker_release_sequences() {
    check_all_outcomes([None, Some(0), Some(1)], || {
        let x = static_atomic(0);
        let y = static_atomic(0);

        let t1 = spawn(move || {
            x.store(2, Relaxed);
        });
        let t2 = spawn(move || {
            y.store(1, Relaxed);
            x.store(1, Release);
            x.store(3, Relaxed);
        });
        let t3 = spawn(move || {
            if x.load(Acquire) == 3 {
                // In C++11, if we read the 3 here, and if the store of 1 was just before the store
                // of 3 in mo order (which it is because we fix the schedule), this forms a release
                // sequence, meaning we acquire the release store of 1, and we can thus never see
                // the value 0.
                // In C++20, this is no longer a release sequence, so 0 can now be observed.
                Some(y.load(Relaxed))
            } else {
                None
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();
        t3.join().unwrap()
    });
}

/// Ensuring normal release sequences (with RMWs) still work correctly.
fn release_sequence() {
    check_all_outcomes([None, Some(1)], || {
        let x = static_atomic(0);
        let y = static_atomic(0);

        let t1 = spawn(move || {
            y.store(1, Relaxed);
            x.store(1, Release);
            x.swap(3, Relaxed);
        });
        let t2 = spawn(move || {
            if x.load(Acquire) == 3 {
                // If we read 3 here, we are seeing the result of the `x.swap` above, which was
                // relaxed but forms a release sequence with the `x.store`. This means there is a
                // release sequence, so we acquire the `y.store` and cannot see the original value
                // `0` any more.
                Some(y.load(Relaxed))
            } else {
                None
            }
        });

        t1.join().unwrap();
        t2.join().unwrap()
    });
}

/// Ensure that when we read from an outdated release store, we acquire its clock.
fn old_release_store() {
    check_all_outcomes([None, Some(1)], || {
        let x = static_atomic(0);
        let y = static_atomic(0);

        let t1 = spawn(move || {
            y.store(1, Relaxed);
            x.store(1, Release); // this is what we want to read from
            x.store(3, Relaxed);
        });
        let t2 = spawn(move || {
            if x.load(Acquire) == 1 {
                // We must have acquired the `y.store` so we cannot see the initial value any more.
                Some(y.load(Relaxed))
            } else {
                None
            }
        });

        t1.join().unwrap();
        t2.join().unwrap()
    });
}

fn main() {
    relaxed();
    seq_cst();
    initialization_write(false);
    initialization_write(true);
    faa_replaced_by_load();
    release_sequence();
    weaker_release_sequences();
    old_release_store();
}
