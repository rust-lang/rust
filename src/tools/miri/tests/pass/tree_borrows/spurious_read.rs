// We ensure a deterministic execution.
// Note that we are *also* using barriers: the barriers enforce the
// specific interleaving of operations that we want, but only the preemption
// rate guarantees that the error message is also deterministic.
//@compile-flags: -Zmiri-deterministic-concurrency
//@compile-flags: -Zmiri-tree-borrows

use std::sync::{Arc, Barrier};
use std::thread;

// A way to send raw pointers across threads.
// Note that when using this in closures will require explicit copying
// `let ptr = ptr;` to force the borrow checker to copy the `Send` wrapper
// instead of just copying the inner `!Send` field.
#[derive(Copy, Clone)]
struct SendPtr(*mut u8);
unsafe impl Send for SendPtr {}

fn main() {
    retagx_retagy_spuriousx_retx_rety_writey();
}

// We're going to enforce a specific interleaving of two
// threads, we use this macro in an effort to make it feasible
// to check in the output that the execution is properly synchronized.
//
// Provide `synchronized!(thread, msg)` where thread is
// a `(thread_id: usize, barrier: Arc<Barrier>)`, and `msg` the message
// to be displayed when the thread reaches this point in the execution.
macro_rules! synchronized {
    ($thread:expr, $msg:expr) => {{
        let (thread_id, barrier) = &$thread;
        eprintln!("Thread {} executing: {}", thread_id, $msg);
        barrier.wait();
    }};
}

// Interleaving:
//   retag x (protect)
//   retag y (protect)
//   spurious read x (target only, which we are executing)
//   ret x
//   ret y
//   write y
//
// This is an interleaving that will never have UB in the source
// (`x` is never accessed for the entire time that `y` is protected).
// For the spurious read to be allowed, we need to check that there is
// no UB in the target (i.e., *with* the spurious read).
//
// The interleaving differs from the one in `tests/fail/tree_borrows/spurious_read.rs` only
// in that it has the `write y` while `y` is no longer protected.
// When the write occurs after protection ends, both source and target are fine
// (checked by this test); when the write occurs during protection, both source
// and target are UB (checked by the `fail` test).
fn retagx_retagy_spuriousx_retx_rety_writey() {
    let mut data = 0u8;
    let ptr = SendPtr(std::ptr::addr_of_mut!(data));
    let barrier = Arc::new(Barrier::new(2));
    let bx = Arc::clone(&barrier);
    let by = Arc::clone(&barrier);

    // This thread only needs to
    // - retag `x` protected
    // - do a read through `x`
    // - remove `x`'s protector
    // Most of the complexity here is synchronization.
    let thread_x = thread::spawn(move || {
        let b = (1, bx);
        synchronized!(b, "start");
        let ptr = ptr;
        synchronized!(b, "retag x (&mut, protect)");
        fn as_mut(x: &mut u8, b: (usize, Arc<Barrier>)) -> *mut u8 {
            synchronized!(b, "retag y (&mut, protect)");
            synchronized!(b, "spurious read x");
            let _v = *x;
            synchronized!(b, "ret x");
            let x = x as *mut u8;
            x
        }
        let _x = as_mut(unsafe { &mut *ptr.0 }, b.clone());
        synchronized!(b, "ret y");
        synchronized!(b, "write y");
        synchronized!(b, "end");
    });

    // This thread's job is to
    // - retag `y` protected
    // - (wait a bit that the other thread performs its spurious read)
    // - remove `y`'s protector
    // - attempt a write through `y`.
    let thread_y = thread::spawn(move || {
        let b = (2, by);
        synchronized!(b, "start");
        let ptr = ptr;
        synchronized!(b, "retag x (&mut, protect)");
        synchronized!(b, "retag y (&mut, protect)");
        fn as_mut(y: &mut u8, b: (usize, Arc<Barrier>)) -> *mut u8 {
            synchronized!(b, "spurious read x");
            synchronized!(b, "ret x");
            let y = y as *mut u8;
            y
        }
        let y = as_mut(unsafe { &mut *ptr.0 }, b.clone());
        synchronized!(b, "ret y");
        synchronized!(b, "write y");
        unsafe {
            *y = 2;
        }
        synchronized!(b, "end");
    });

    thread_x.join().unwrap();
    thread_y.join().unwrap();
}
