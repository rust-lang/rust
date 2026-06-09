//@ignore-target: windows # No pthreads on Windows
//
// Check that if we do not set the mutex type, it is UB to do reentrant locking. glibc apparently
// actually exploits this, see
// <https://github.molgen.mpg.de/git-mirror/glibc/blob/master/nptl/pthread_mutexattr_settype.c#L31>:
// one must actively call pthread_mutexattr_settype to disable lock elision. This means a call to
// pthread_mutexattr_settype(PTHREAD_MUTEX_NORMAL) makes a difference even if
// PTHREAD_MUTEX_NORMAL == PTHREAD_MUTEX_DEFAULT!

fn main() {
    unsafe {
        let mut mutexattr: libc::pthread_mutexattr_t = std::mem::zeroed();
        assert_eq!(libc::pthread_mutexattr_init(&mut mutexattr as *mut _), 0);
        let mut mutex: libc::pthread_mutex_t = std::mem::zeroed();
        assert_eq!(libc::pthread_mutex_init(&mut mutex as *mut _, &mutexattr as *const _), 0);
        assert_eq!(libc::pthread_mutex_lock(&mut mutex as *mut _), 0);
        libc::pthread_mutex_lock(&mut mutex as *mut _); //~ ERROR: already locked by the current thread
    }
}
