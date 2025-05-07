//@ignore-target: windows # No pthreads on Windows

/// Test that destroying a pthread_cond twice fails, even without a check for number validity

fn main() {
    unsafe {
        use core::mem::MaybeUninit;
        let mut attr = MaybeUninit::<libc::pthread_condattr_t>::uninit();
        libc::pthread_condattr_init(attr.as_mut_ptr());

        let mut cond = MaybeUninit::<libc::pthread_cond_t>::uninit();

        libc::pthread_cond_init(cond.as_mut_ptr(), attr.as_ptr());

        libc::pthread_cond_destroy(cond.as_mut_ptr());

        libc::pthread_cond_destroy(cond.as_mut_ptr());
        //~^ ERROR: Undefined Behavior: using uninitialized data, but this operation requires initialized memory
    }
}
