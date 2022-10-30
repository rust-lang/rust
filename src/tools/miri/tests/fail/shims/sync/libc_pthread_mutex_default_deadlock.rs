//@ignore-target-windows: No libc on Windows
//
// Check that if we do not set the mutex type, it is the default.

fn main() {
    unsafe {
        let mutexattr: libc::pthread_mutexattr_t = std::mem::zeroed();
        let mut mutex: libc::pthread_mutex_t = std::mem::zeroed();
        assert_eq!(libc::pthread_mutex_init(&mut mutex as *mut _, &mutexattr as *const _), 0);
        assert_eq!(libc::pthread_mutex_lock(&mut mutex as *mut _), 0);
        libc::pthread_mutex_lock(&mut mutex as *mut _); //~ ERROR: Undefined Behavior: trying to acquire already locked default mutex
    }
}
