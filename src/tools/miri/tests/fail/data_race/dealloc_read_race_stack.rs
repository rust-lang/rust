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
    //  2. read
    //  3. stack-deallocate
    unsafe {
        let j1 = spawn(move || {
            let pointer = &*ptr.0;
            {
                let mut stack_var = 0usize;

                pointer.store(&mut stack_var as *mut _, Ordering::Release);

                sleep(Duration::from_millis(200));

                // Now `stack_var` gets deallocated.
            } //~ ERROR: Data race detected between Deallocate on thread `<unnamed>` and Read on thread `<unnamed>`
        });

        let j2 = spawn(move || {
            let pointer = &*ptr.0;
            *pointer.load(Ordering::Acquire)
        });

        j1.join().unwrap();
        j2.join().unwrap();
    }
}
