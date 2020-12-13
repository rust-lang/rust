// ignore-windows: Concurrency on Windows is not supported yet.

use std::thread::spawn;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

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
            let atomic_ref = &mut *c.0;
            atomic_ref.store(64, Ordering::SeqCst);
        });

        let j2 = spawn(move || {
            let atomic_ref = &mut *c.0;
            *atomic_ref.get_mut() = 32; //~ ERROR Data race detected between Write on Thread(id = 2) and Atomic Store on Thread(id = 1)
        });

        j1.join().unwrap();
        j2.join().unwrap();
    }
}
