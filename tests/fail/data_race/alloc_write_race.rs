// ignore-windows: Concurrency on Windows is not supported yet.
#![feature(new_uninit)]

use std::thread::spawn;
use std::ptr::null_mut;
use std::sync::atomic::{Ordering, AtomicPtr};

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
            // Concurrent allocate the memory.
            // Uses relaxed semantics to not generate
            // a release sequence.
            let pointer = &*ptr.0;
            pointer.store(Box::into_raw(Box::<usize>::new_uninit()) as *mut usize, Ordering::Relaxed);
        });

        let j2 = spawn(move || {
            let pointer = &*ptr.0;
            *pointer.load(Ordering::Relaxed) = 2; //~ ERROR Data race detected between Write on Thread(id = 2) and Allocate on Thread(id = 1)
        });

        j1.join().unwrap();
        j2.join().unwrap();

        // Clean up memory, will never be executed
        drop(Box::from_raw(pointer.load(Ordering::Relaxed)));
    }
}
