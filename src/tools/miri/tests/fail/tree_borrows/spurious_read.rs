// We ensure a deterministic execution.
// Note that we are *also* using barriers: the barriers enforce the
// specific interleaving of operations that we want, but we need to disable
// preemption to ensure that the error message is also deterministic.
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
    retagx_retagy_retx_writey_rety();
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
//   spurious read x (target only, which we are *not* executing)
//   ret x
//   write y
//   ret y
//
// This is an interleaving that will never *not* have UB in the target
// (`noalias` violation on `y`).
// For the spurious read to be allowed, we need to ensure there *is* UB
// in the source (i.e., without the spurious read).
//
// The interleaving differs from the one in `tests/pass/tree_borrows/spurious_read.rs` only
// in that it has the `write y` while `y` is still protected.
// When the write occurs after protection ends, both source and target are fine
// (checked by the `pass` test); when the write occurs during protection, both source
// and target are UB (checked by this test).
fn retagx_retagy_retx_writey_rety() {
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
            synchronized!(b, "location where spurious read of x would happen in the target");
            // This is ensuring taht we have UB *without* the spurious read,
            // so we don't read here.
            synchronized!(b, "ret x");
            synchronized!(b, "write y");
            let x = x as *mut u8;
            x
        }
        let _x = as_mut(unsafe { &mut *ptr.0 }, b.clone());
        synchronized!(b, "ret y");
        synchronized!(b, "end");
    });

    // This thread's job is to
    // - retag `y` protected
    // - (wait for the other thread to return so that there is no foreign protector when we write)
    // - attempt a write through `y`.
    // - (UB should have occurred by now, but the next step would be to
    //    remove `y`'s protector)
    let thread_y = thread::spawn(move || {
        let b = (2, by);
        synchronized!(b, "start");
        let ptr = ptr;
        synchronized!(b, "retag x (&mut, protect)");
        synchronized!(b, "retag y (&mut, protect)");
        fn as_mut(y: &mut u8, b: (usize, Arc<Barrier>)) -> *mut u8 {
            synchronized!(b, "location where spurious read of x would happen in the target");
            synchronized!(b, "ret x");
            let y = y as *mut u8;
            synchronized!(b, "write y");
            unsafe {
                *y = 2; //~ERROR: /write access through .* is forbidden/
            }
            synchronized!(b, "ret y");
            y
        }
        let _y = as_mut(unsafe { &mut *ptr.0 }, b.clone());
        synchronized!(b, "end");
    });

    thread_x.join().unwrap();
    thread_y.join().unwrap();
}
