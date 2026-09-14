// This is mostly the same test as `tests/pass/tree_borrows/retag_no_race.rs`, but it now fails under implicit writes, meaning that there is now Undefined Behaviour. It has been changed slightly have the same interleaving on all targets.
//@compile-flags: -Zmiri-tree-borrows -Zmiri-tree-borrows-implicit-writes
// This test relies on a specific interleaving that cannot be enforced with just barriers. We must remove preemption so that the execution and the error messages are deterministic.
//@compile-flags: -Zmiri-deterministic-concurrency
use std::ptr::addr_of_mut;
use std::sync::{Arc, Barrier};
use std::thread;

#[derive(Copy, Clone)]
struct SendPtr(*mut u8);

unsafe impl Send for SendPtr {}

fn thread_1(x: SendPtr, barrier: Arc<Barrier>) {
    let x = unsafe { &mut *x.0 };
    barrier.wait(); // init

    let _v = *x;

    barrier.wait(); // write
}

fn thread_2(y: SendPtr, barrier: Arc<Barrier>) {
    let y = unsafe { &mut *y.0 };
    barrier.wait(); // init
    thread::yield_now(); // force other thread to read first

    fn write(y: &mut u8, v: u8, barrier: &Arc<Barrier>) {
        //~^ ERROR: /Undefined Behavior: Data race detected between .* non-atomic read on thread .* and .* retag write of type .* on thread .* at .*/
        barrier.wait(); // write
        *y = v;
    }
    write(&mut *y, 42, &barrier);
}

fn main() {
    let mut data = 0u8;
    let p = SendPtr(addr_of_mut!(data));
    let barrier = Arc::new(Barrier::new(2));
    let b1 = Arc::clone(&barrier);
    let b2 = Arc::clone(&barrier);

    let h1 = thread::spawn(move || thread_1(p, b1));
    let h2 = thread::spawn(move || thread_2(p, b2));
    h1.join().unwrap();
    h2.join().unwrap();
}
