//@ignore-target: windows # No pthreads on Windows

fn main() {
    unsafe {
        let mut mutex: libc::pthread_mutex_t = std::mem::zeroed();
        assert_eq!(libc::pthread_mutex_init(&mut mutex as *mut _, std::ptr::null()), 0);
        assert_eq!(libc::pthread_mutex_lock(&mut mutex as *mut _), 0);
        libc::pthread_mutex_destroy(&mut mutex as *mut _); //~ ERROR: destroyed a locked mutex
    }
}
