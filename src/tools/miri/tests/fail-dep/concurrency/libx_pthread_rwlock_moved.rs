//@ignore-target: windows # No pthreads on Windows

fn main() {
    unsafe {
        let mut rw = libc::PTHREAD_RWLOCK_INITIALIZER;

        libc::pthread_rwlock_rdlock(&mut rw as *mut _);

        // Move rwlock
        let mut rw2 = rw;

        libc::pthread_rwlock_unlock(&mut rw2 as *mut _); //~ ERROR: can't be moved after first use
    }
}
