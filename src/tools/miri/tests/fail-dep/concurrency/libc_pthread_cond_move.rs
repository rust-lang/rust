//@revisions: static_initializer init
//@ignore-target: windows # No pthreads on Windows

/// Test that moving a pthread_cond between uses fails.

fn main() {
    check()
}

#[cfg(init)]
fn check() {
    unsafe {
        use core::mem::MaybeUninit;
        let mut cond = MaybeUninit::<libc::pthread_cond_t>::uninit();

        libc::pthread_cond_init(cond.as_mut_ptr(), std::ptr::null());

        // move pthread_cond_t
        let mut cond2 = cond;

        libc::pthread_cond_destroy(cond2.as_mut_ptr()); //~[init] ERROR: can't be moved after first use
    }
}

#[cfg(static_initializer)]
fn check() {
    unsafe {
        let mut cond = libc::PTHREAD_COND_INITIALIZER;

        libc::pthread_cond_signal(&mut cond as *mut _);

        // move pthread_cond_t
        let mut cond2 = cond;

        libc::pthread_cond_destroy(&mut cond2 as *mut _); //~[static_initializer] ERROR: can't be moved after first use
    }
}
