//@compile-flags: -Zmiri-disable-isolation -Zmiri-disable-weak-memory-emulation -Zmiri-preemption-rate=0

use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::thread::{sleep, spawn};
use std::time::Duration;

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
    //  1. stack-allocate
    //  2. atomic_store
    //  3. atomic_load
    //  4. write-value
    //  5. write-value
    unsafe {
        let j1 = spawn(move || {
            // Concurrent allocate the memory.
            // Uses relaxed semantics to not generate
            // a release sequence.
            let pointer = &*ptr.0;

            let mut stack_var = 0usize;

            pointer.store(&mut stack_var as *mut _, Ordering::Release);

            sleep(Duration::from_millis(200));

            stack_var = 1usize; //~ ERROR: Data race detected between Write on thread `<unnamed>` and Write on thread `<unnamed>`

            // read to silence errors
            stack_var
        });

        let j2 = spawn(move || {
            let pointer = &*ptr.0;
            *pointer.load(Ordering::Acquire) = 3;
        });

        j1.join().unwrap();
        j2.join().unwrap();
    }
}
