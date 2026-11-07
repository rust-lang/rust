//@ignore-target: windows # No pthreads on Windows
use std::mem;

fn main() {
    unsafe {
        let mutexattr: libc::pthread_mutexattr_t = mem::zeroed();
        let mut mutex: libc::pthread_mutex_t = mem::zeroed();
        assert_eq!(libc::pthread_mutex_init(&mut mutex as *mut _, &mutexattr as *const _), 0);
        //~^ERROR: has not been initialized
    }
}
