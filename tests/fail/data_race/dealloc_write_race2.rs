// We want to control preemption here.
//@compile-flags: -Zmiri-preemption-rate=0

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
            __rust_dealloc(
                ptr.0 as *mut _,
                std::mem::size_of::<usize>(),
                std::mem::align_of::<usize>(),
            );
        });

        let j2 = spawn(move || {
            // Also an error of the form: Data race detected between Write on thread `<unnamed>` and Deallocate on thread `<unnamed>`
            // but the invalid allocation is detected first.
            *ptr.0 = 2; //~ ERROR: dereferenced after this allocation got freed
        });

        j1.join().unwrap();
        j2.join().unwrap();
    }
}
