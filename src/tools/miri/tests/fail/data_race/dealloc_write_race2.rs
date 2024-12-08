// We want to control preemption here. Stacked borrows interferes by having its own accesses.
//@compile-flags: -Zmiri-preemption-rate=0 -Zmiri-disable-stacked-borrows
// Avoid accidental synchronization via address reuse inside `thread::spawn`.
//@compile-flags: -Zmiri-address-reuse-cross-thread-rate=0

use std::thread::spawn;

#[derive(Copy, Clone)]
struct EvilSend<T>(pub T);

unsafe impl<T> Send for EvilSend<T> {}
unsafe impl<T> Sync for EvilSend<T> {}

extern "Rust" {
    fn __rust_dealloc(ptr: *mut u8, size: usize, align: usize);
}
pub fn main() {
    // Shared atomic pointer
    let pointer: *mut usize = Box::into_raw(Box::new(0usize));
    let ptr = EvilSend(pointer);

    unsafe {
        let j1 = spawn(move || {
            let ptr = ptr; // avoid field capturing
            __rust_dealloc(
                ptr.0 as *mut _,
                std::mem::size_of::<usize>(),
                std::mem::align_of::<usize>(),
            );
        });

        let j2 = spawn(move || {
            let ptr = ptr; // avoid field capturing
            // Also an error of the form: Data race detected between (1) deallocation on thread `unnamed-ID` and (2) non-atomic write on thread `unnamed-ID`
            // but the invalid allocation is detected first.
            *ptr.0 = 2; //~ ERROR: has been freed
        });

        j1.join().unwrap();
        j2.join().unwrap();
    }
}
