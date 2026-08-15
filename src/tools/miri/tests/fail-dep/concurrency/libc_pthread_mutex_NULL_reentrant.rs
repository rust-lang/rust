//@ignore-target: windows # No pthreads on Windows
//
// Check that if we pass NULL attribute, then reentrant locking is UB.

fn main() {
    unsafe {
        let mut mutex: libc::pthread_mutex_t = std::mem::zeroed();
        assert_eq!(libc::pthread_mutex_init(&mut mutex as *mut _, std::ptr::null() as *const _), 0);
        assert_eq!(libc::pthread_mutex_lock(&mut mutex as *mut _), 0);
        libc::pthread_mutex_lock(&mut mutex as *mut _); //~ ERROR: already locked by the current thread
    }
}
