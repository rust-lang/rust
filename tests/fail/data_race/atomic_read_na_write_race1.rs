// ignore-windows: Concurrency on Windows is not supported yet.
#![feature(core_intrinsics)]

use std::thread::spawn;
use std::sync::atomic::AtomicUsize;
use std::intrinsics::atomic_load;

#[derive(Copy, Clone)]
struct EvilSend<T>(pub T);

unsafe impl<T> Send for EvilSend<T> {}
unsafe impl<T> Sync for EvilSend<T> {}

pub fn main() {
    let mut a = AtomicUsize::new(0);
    let b = &mut a as *mut AtomicUsize;
    let c = EvilSend(b);
    unsafe {
        let j1 = spawn(move || {
            *(c.0 as *mut usize) = 32;
        });

        let j2 = spawn(move || {
            //Equivalent to: (&*c.0).load(Ordering::SeqCst)
            atomic_load(c.0 as *mut usize) //~ ERROR Data race detected between Atomic Load on Thread(id = 2) and Write on Thread(id = 1)
        });

        j1.join().unwrap();
        j2.join().unwrap();
    }
}
