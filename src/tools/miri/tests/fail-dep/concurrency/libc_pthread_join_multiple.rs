//@ignore-target: windows # No pthreads on Windows

// Joining the same thread from multiple threads is undefined behavior.

use std::{mem, ptr, thread};

extern "C" fn thread_start(_null: *mut libc::c_void) -> *mut libc::c_void {
    // Yield the thread several times so that other threads can join it.
    thread::yield_now();
    thread::yield_now();
    ptr::null_mut()
}

fn main() {
    unsafe {
        let mut native: libc::pthread_t = mem::zeroed();
        assert_eq!(
            libc::pthread_create(&mut native, ptr::null(), thread_start, ptr::null_mut()),
            0
        );
        let mut native_copy: libc::pthread_t = mem::zeroed();
        ptr::copy_nonoverlapping(&native, &mut native_copy, 1);
        let handle = thread::spawn(move || {
            assert_eq!(libc::pthread_join(native_copy, ptr::null_mut()), 0); //~ ERROR: Undefined Behavior: trying to join an already joined thread
        });
        assert_eq!(libc::pthread_join(native, ptr::null_mut()), 0);
        handle.join().unwrap();
    }
}
