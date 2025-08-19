//@compile-flags: -Zmiri-tree-borrows
// This test relies on a specific interleaving that cannot be enforced
// with just barriers. We must remove preemption so that the execution and the
// error messages are deterministic.
//@compile-flags: -Zmiri-deterministic-concurrency
use std::ptr::addr_of_mut;
use std::sync::{Arc, Barrier};
use std::thread;

#[derive(Copy, Clone)]
struct SendPtr(*mut u8);

unsafe impl Send for SendPtr {}

// This test features the problematic pattern
//
// read x     || retag y (&mut, protect)
//        -- sync --
//            || write y
//
// In which
// - one interleaving (`1:read; 2:retag; 2:write`) does not have UB if retags
//   count only as reads for the data race model,
// - the other interleaving (`2:retag; 1:read; 2:write`) has UB (`noalias` violation).
//
// The interleaving executed here is the one that does not have UB,
// i.e.
//      1:read x
//      2:retag y
//      2:write y
//
// Tree Borrows considers that the read of `x` cannot be in conflict
// with `y` because `y` did not even exist yet when `x` was accessed.
//
// As long as we are not emitting any writes for the data race model
// upon retags of mutable references, it should not have any issue with
// this code either.
// We do not want to emit a write for the data race model, because
// although there is race-like behavior going on in this pattern
// (where some but not all interleavings contain UB), making this an actual
// data race has the confusing consequence of one single access being treated
// as being of different `AccessKind`s by different parts of Miri
// (a retag would be always a read for the aliasing model, and sometimes a write
// for the data race model).

// The other interleaving is a subsequence of `tests/fail/tree_borrows/spurious_read.rs`
// which asserts that
//      2:retag y
//      1:read x
//      2:write y
// is UB.

type IdxBarrier = (usize, Arc<Barrier>);
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

fn thread_1(x: SendPtr, barrier: IdxBarrier) {
    let x = unsafe { &mut *x.0 };
    synchronized!(barrier, "spawn");

    synchronized!(barrier, "read x || retag y");
    // This is the interleaving without UB: by the time
    // the other thread starts retagging, this thread
    // has already finished all its work using `y`.
    let _v = *x;
    synchronized!(barrier, "write y");
    synchronized!(barrier, "exit");
}

fn thread_2(y: SendPtr, barrier: IdxBarrier) {
    let y = unsafe { &mut *y.0 };
    synchronized!(barrier, "spawn");

    fn write(y: &mut u8, v: u8, barrier: &IdxBarrier) {
        synchronized!(barrier, "write y");
        *y = v;
    }
    synchronized!(barrier, "read x || retag y");
    // We don't use a barrier here so that *if* the retag counted as a write
    // for the data race model, then it would be UB.
    // We still want to make sure that the other thread goes first as per the
    // interleaving that we are testing, so we use `yield_now + preemption-rate=0`
    // which has the effect of forcing a specific interleaving while still
    // not counting as "synchronization" from the point of view of the data
    // race model.
    thread::yield_now();
    write(&mut *y, 42, &barrier);
    synchronized!(barrier, "exit");
}

fn main() {
    let mut data = 0u8;
    let p = SendPtr(addr_of_mut!(data));
    let barrier = Arc::new(Barrier::new(2));
    let b1 = (1, Arc::clone(&barrier));
    let b2 = (2, Arc::clone(&barrier));

    let h1 = thread::spawn(move || thread_1(p, b1));
    let h2 = thread::spawn(move || thread_2(p, b2));
    h1.join().unwrap();
    h2.join().unwrap();
}
