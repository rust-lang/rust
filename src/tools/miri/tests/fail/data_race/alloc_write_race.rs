// We want to control preemption here. Stacked borrows interferes by having its own accesses.
//@compile-flags: -Zmiri-deterministic-concurrency -Zmiri-disable-stacked-borrows

use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::thread::spawn;

#[derive(Copy, Clone)]
struct EvilSend<T>(pub T);

unsafe impl<T> Send for EvilSend<T> {}
unsafe impl<T> Sync for EvilSend<T> {}

pub fn main() {
    // Shared atomic pointer
    let pointer = AtomicPtr::new(null_mut::<usize>());
    let ptr = EvilSend(&pointer as *const AtomicPtr<usize>);

    // Note: this is scheduler-dependent
    // the operations need to occur in
    // order, otherwise the allocation is
    // not visible to the other-thread to
    // detect the race:
    //  1. alloc
    //  2. write
    unsafe {
        let j1 = spawn(move || {
            let ptr = ptr; // avoid field capturing
            // Concurrent allocate the memory.
            // Uses relaxed semantics to not generate
            // a release sequence.
            let pointer = &*ptr.0;
            pointer
                .store(Box::into_raw(Box::<usize>::new_uninit()) as *mut usize, Ordering::Relaxed);
        });

        let j2 = spawn(move || {
            let ptr = ptr; // avoid field capturing
            let pointer = &*ptr.0;
            *pointer.load(Ordering::Relaxed) = 2; //~ ERROR: Data race detected between (1) creating a new allocation on thread `unnamed-1` and (2) non-atomic write on thread `unnamed-2`
        });

        j1.join().unwrap();
        j2.join().unwrap();

        // Clean up memory, will never be executed
        drop(Box::from_raw(pointer.load(Ordering::Relaxed)));
    }
}
