//@ignore-target: windows # No pthreads on Windows
//
// Check that if we use PTHREAD_MUTEX_INITIALIZER, then reentrant locking is UB.
// glibc apparently actually exploits this so we better catch it!

fn main() {
    unsafe {
        let mut mutex: libc::pthread_mutex_t = libc::PTHREAD_MUTEX_INITIALIZER;
        assert_eq!(libc::pthread_mutex_lock(&mut mutex as *mut _), 0);
        libc::pthread_mutex_lock(&mut mutex as *mut _); //~ ERROR: already locked by the current thread
    }
}
