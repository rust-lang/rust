//@ignore-target: windows # No pthreads on Windows

fn main() {
    unsafe {
        let mut mutexattr: libc::pthread_mutexattr_t = std::mem::zeroed();
        assert_eq!(
            libc::pthread_mutexattr_settype(&mut mutexattr as *mut _, libc::PTHREAD_MUTEX_NORMAL),
            0,
        );
        let mut mutex: libc::pthread_mutex_t = std::mem::zeroed();
        assert_eq!(libc::pthread_mutex_init(&mut mutex as *mut _, &mutexattr as *const _), 0);
        assert_eq!(libc::pthread_mutex_lock(&mut mutex as *mut _), 0);
        // A "normal" mutex properly tries to acquire the lock even if its is already held
        // by the current thread -- and then we deadlock.
        libc::pthread_mutex_lock(&mut mutex as *mut _); //~ ERROR: the evaluated program deadlocked
    }
}
