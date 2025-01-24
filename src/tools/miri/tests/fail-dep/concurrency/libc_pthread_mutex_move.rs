//@ignore-target: windows # No pthreads on Windows
//@revisions: static_initializer init

fn main() {
    check();
}

#[cfg(init)]
fn check() {
    unsafe {
        let mut m: libc::pthread_mutex_t = std::mem::zeroed();
        assert_eq!(libc::pthread_mutex_init(&mut m as *mut _, std::ptr::null()), 0);

        let mut m2 = m; // move the mutex
        libc::pthread_mutex_lock(&mut m2 as *mut _); //~[init] ERROR: can't be moved after first use
    }
}

#[cfg(static_initializer)]
fn check() {
    unsafe {
        let mut m: libc::pthread_mutex_t = libc::PTHREAD_MUTEX_INITIALIZER;
        libc::pthread_mutex_lock(&mut m as *mut _);

        let mut m2 = m; // move the mutex
        libc::pthread_mutex_unlock(&mut m2 as *mut _); //~[static_initializer] ERROR: can't be moved after first use
    }
}
