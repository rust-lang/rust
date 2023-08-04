//@revisions: stack tree
//@compile-flags: -Zmiri-preemption-rate=0
//@[tree]compile-flags: -Zmiri-tree-borrows
use std::thread;

#[derive(Copy, Clone)]
struct SendPtr(*mut i32);
unsafe impl Send for SendPtr {}

fn main() {
    let mut mem = 0;
    let ptr = SendPtr(&mut mem as *mut _);

    let t = thread::spawn(move || {
        let ptr = ptr;
        // We do a protected 2phase retag (but no write!) in this thread.
        fn retag(_x: &mut i32) {} //~[tree]ERROR: Data race detected between (1) Read on thread `main` and (2) Write on thread `<unnamed>`
        retag(unsafe { &mut *ptr.0 }); //~[stack]ERROR: Data race detected between (1) Read on thread `main` and (2) Write on thread `<unnamed>`
    });

    // We do a read in the main thread.
    unsafe { ptr.0.read() };

    // These two operations do not commute -- if the read happens after the retag, the retagged pointer
    // gets frozen! So we want this to be considered UB so that we can still freely move the read around
    // in this thread without worrying about reordering with retags in other threads.

    t.join().unwrap();
}
