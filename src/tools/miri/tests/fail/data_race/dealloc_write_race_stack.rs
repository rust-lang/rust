// We want to control preemption here. Stacked borrows interferes by having its own accesses.
//@compile-flags: -Zmiri-deterministic-concurrency -Zmiri-disable-stacked-borrows

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
    //  2. read
    //  3. stack-deallocate
    unsafe {
        let j1 = spawn(move || {
            let ptr = ptr; // avoid field capturing
            let pointer = &*ptr.0;
            {
                let mut stack_var = 0usize;

                pointer.store(&mut stack_var as *mut _, Ordering::Release);

                sleep(Duration::from_millis(200));

                // Now `stack_var` gets deallocated.
            } //~ ERROR: Data race detected between (1) non-atomic write on thread `unnamed-2` and (2) deallocation on thread `unnamed-1`
        });

        let j2 = spawn(move || {
            let ptr = ptr; // avoid field capturing
            let pointer = &*ptr.0;
            *pointer.load(Ordering::Acquire) = 3;
        });

        j1.join().unwrap();
        j2.join().unwrap();
    }
}
