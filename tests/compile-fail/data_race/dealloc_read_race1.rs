// ignore-windows: Concurrency on Windows is not supported yet.

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
            *ptr.0
        });

        let j2 = spawn(move || {
            __rust_dealloc(ptr.0 as *mut _, std::mem::size_of::<usize>(), std::mem::align_of::<usize>());  //~ ERROR Data race detected between Deallocate on Thread(id = 2) and Read on Thread(id = 1)
        });

        j1.join().unwrap();
        j2.join().unwrap();
    }
}
