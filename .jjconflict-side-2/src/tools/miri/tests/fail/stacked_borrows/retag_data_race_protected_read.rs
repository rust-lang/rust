//@compile-flags:-Zmiri-deterministic-concurrency
use std::thread;

#[derive(Copy, Clone)]
struct SendPtr(*mut i32);
unsafe impl Send for SendPtr {}

fn main() {
    let mut mem = 0;
    let ptr = SendPtr(&mut mem as *mut _);

    let t = thread::spawn(move || {
        let ptr = ptr;
        // We do a protected mutable retag (but no write!) in this thread.
        fn retag(_x: &mut i32) {}
        retag(unsafe { &mut *ptr.0 }); //~ERROR: Data race detected between (1) non-atomic read on thread `main` and (2) retag write of type `i32` on thread `unnamed-1`
    });

    // We do a read in the main thread.
    unsafe { ptr.0.read() };

    // These two operations do not commute!
    // - In Stacked Borrows, if the read happens after the retag it will `Disable` the pointer.
    // - In Tree Borrows, if the read happens after the retag, the retagged pointer gets frozen!
    // Ideally we would want this to be considered UB so that we can still freely move the read around
    // in this thread without worrying about reordering with retags in other threads,
    // but in Tree Borrows we have found worse issues that occur if we make this a data race.

    t.join().unwrap();
}
