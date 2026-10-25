//@ignore-target: windows # No pthreads on Windows
//@ignore-target: apple # nothing to store in pthread_condattr_t
use std::mem;

fn main() {
    unsafe {
        let condattr: libc::pthread_condattr_t = mem::zeroed();
        let mut cond: libc::pthread_cond_t = mem::zeroed();
        assert_eq!(libc::pthread_cond_init(&mut cond as *mut _, &condattr as *const _), 0);
        //~^ERROR: not initialized
    }
}
