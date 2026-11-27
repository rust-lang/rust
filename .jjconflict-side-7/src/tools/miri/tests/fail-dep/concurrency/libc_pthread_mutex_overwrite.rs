//@ignore-target: windows # No pthreads on Windows

fn main() {
    unsafe {
        let mut m: libc::pthread_mutex_t = libc::PTHREAD_MUTEX_INITIALIZER;
        libc::pthread_mutex_lock(&mut m as *mut _);

        // Overwrite the mutex with itself. This de-initializes it.
        let copy = m;
        m = copy;

        libc::pthread_mutex_unlock(&mut m as *mut _); //~ERROR: not properly initialized
    }
}
