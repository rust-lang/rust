//@compile-flags: -Zmiri-disable-data-race-detector
// Avoid non-determinism
//@compile-flags: -Zmiri-deterministic-concurrency

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
            let c = c; // avoid field capturing
            *c.0 = 32;
        });

        let j2 = spawn(move || {
            let c = c; // avoid field capturing
            *c.0 = 64; // Data race (but not detected as the detector is disabled)
        });

        j1.join().unwrap();
        j2.join().unwrap();
    }
}
