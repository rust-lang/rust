// Illustrating a problematic interaction between Reserved, interior mutability,
// and protectors, that makes spurious writes fail in the previous model of Tree Borrows.
// As for all similar tests, we disable preemption so that the error message is deterministic.
//@compile-flags: -Zmiri-tree-borrows -Zmiri-deterministic-concurrency
//
// One revision without spurious read (default source code) and one with spurious read.
// Both are expected to be UB. Both revisions are expected to have the *same* error
// because we are aligning the behavior of `without` to that of `with` so that the
// spurious write is effectively a noop in the long term.
//@revisions: without with

use std::cell::Cell;
use std::sync::{Arc, Barrier};
use std::thread;

// Here is the problematic interleaving:
// - thread 1: retag and activate `x` (protected)
// - thread 2: retag but do not initialize (lazy) `y` as Reserved with interior mutability
// - thread 1: spurious write through `x` would go here
// - thread 2: function exit (noop due to lazyness)
// - thread 1: function exit (no permanent effect on `y` because it is now Reserved IM unprotected)
// - thread 2: write through `y`
// In the source code nothing happens to `y`

// `Send`able raw pointer wrapper.
#[derive(Copy, Clone)]
struct SendPtr(*mut u8);
unsafe impl Send for SendPtr {}

type IdxBarrier = (usize, Arc<Barrier>);

// Barriers to enforce the interleaving.
// This macro expects `synchronized!(thread, msg)` where `thread` is a `IdxBarrier`,
// and `msg` is the message to be displayed when the thread reaches this point in the execution.
macro_rules! synchronized {
    ($thread:expr, $msg:expr) => {{
        let (thread_id, barrier) = &$thread;
        eprintln!("Thread {} executing: {}", thread_id, $msg);
        barrier.wait();
    }};
}

fn main() {
    // The conflict occurs on one single location but the example involves
    // lazily initialized permissions. We will use `&mut Cell<()>` references
    // to `data` to achieve this.
    let mut data = 0u8;
    let ptr = SendPtr(std::ptr::addr_of_mut!(data));
    let barrier = Arc::new(Barrier::new(2));
    let bx = Arc::clone(&barrier);
    let by = Arc::clone(&barrier);

    // Retag and activate `x`, wait until the other thread creates a lazy permission.
    // Then do a spurious write. Finally exit the function after the other thread.
    let thread_1 = thread::spawn(move || {
        let b = (1, bx);
        synchronized!(b, "start");
        let ptr = ptr;
        synchronized!(b, "retag x (&mut, protect)");
        fn inner(x: &mut u8, b: IdxBarrier) {
            *x = 42; // activate immediately
            synchronized!(b, "[lazy] retag y (&mut, protect, IM)");
            // A spurious write should be valid here because `x` is
            // `Active` and protected.
            if cfg!(with) {
                synchronized!(b, "spurious write x (executed)");
                *x = 64;
            } else {
                synchronized!(b, "spurious write x (skipped)");
            }
            synchronized!(b, "ret y");
            synchronized!(b, "ret x");
        }
        inner(unsafe { &mut *ptr.0 }, b.clone());
        synchronized!(b, "write y");
        synchronized!(b, "end");
    });

    // Create a lazy Reserved with interior mutability.
    // Wait for the other thread's spurious write then observe the side effects
    // of that write.
    let thread_2 = thread::spawn(move || {
        let b = (2, by);
        synchronized!(b, "start");
        let ptr = ptr;
        synchronized!(b, "retag x (&mut, protect)");
        synchronized!(b, "[lazy] retag y (&mut, protect, IM)");
        fn inner(y: &mut Cell<()>, b: IdxBarrier) -> *mut u8 {
            synchronized!(b, "spurious write x");
            synchronized!(b, "ret y");
            // `y` is not retagged for any bytes, so the pointer we return
            // has its permission lazily initialized.
            y as *mut Cell<()> as *mut u8
        }
        // Currently `ptr` points to `data`.
        // We do a zero-sized retag so that its permission is lazy.
        let y_zst = unsafe { &mut *(ptr.0 as *mut Cell<()>) };
        let y = inner(y_zst, b.clone());
        synchronized!(b, "ret x");
        synchronized!(b, "write y");
        unsafe { *y = 13 } //~ERROR: /write access through .* is forbidden/
        synchronized!(b, "end");
    });

    thread_1.join().unwrap();
    thread_2.join().unwrap();
}
