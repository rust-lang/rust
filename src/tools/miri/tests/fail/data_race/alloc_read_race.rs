//@compile-flags: -Zmiri-disable-weak-memory-emulation -Zmiri-preemption-rate=0
#![feature(new_uninit)]

use std::mem::MaybeUninit;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::thread::spawn;

#[derive(Copy, Clone)]
struct EvilSend<T>(pub T);

unsafe impl<T> Send for EvilSend<T> {}
unsafe impl<T> Sync for EvilSend<T> {}

pub fn main() {
    // Shared atomic pointer
    let pointer = AtomicPtr::new(null_mut::<MaybeUninit<usize>>());
    let ptr = EvilSend(&pointer as *const AtomicPtr<MaybeUninit<usize>>);

    // Note: this is scheduler-dependent
    // the operations need to occur in
    // order, otherwise the allocation is
    // not visible to the other-thread to
    // detect the race:
    //  1. alloc
    //  2. write
    unsafe {
        let j1 = spawn(move || {
            // Concurrent allocate the memory.
            // Uses relaxed semantics to not generate
            // a release sequence.
            let pointer = &*ptr.0;
            pointer.store(Box::into_raw(Box::new_uninit()), Ordering::Relaxed);
        });

        let j2 = spawn(move || {
            let pointer = &*ptr.0;

            // Note: could also error due to reading uninitialized memory, but the data-race detector triggers first.
            *pointer.load(Ordering::Relaxed) //~ ERROR: Data race detected between Read on thread `<unnamed>` and Allocate on thread `<unnamed>`
        });

        j1.join().unwrap();
        j2.join().unwrap();

        // Clean up memory, will never be executed
        drop(Box::from_raw(pointer.load(Ordering::Relaxed)));
    }
}
