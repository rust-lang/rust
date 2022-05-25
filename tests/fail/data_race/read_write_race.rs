// We want to control preemption here.
//@compile-flags: -Zmiri-preemption-rate=0

use std::thread::spawn;

#[derive(Copy, Clone)]
struct EvilSend<T>(pub T);

unsafe impl<T> Send for EvilSend<T> {}
unsafe impl<T> Sync for EvilSend<T> {}

pub fn main() {
    let mut a = 0u32;
    let b = &mut a as *mut u32;
    let c = EvilSend(b);
    unsafe {
        let j1 = spawn(move || {
            let _val = *c.0;
        });

        let j2 = spawn(move || {
            *c.0 = 64; //~ ERROR: Data race detected between Write on thread `<unnamed>` and Read on thread `<unnamed>`
        });

        j1.join().unwrap();
        j2.join().unwrap();
    }
}
