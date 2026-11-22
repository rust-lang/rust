//@ignore-target: windows # No pthreads on Windows

/// Test that destroying a pthread_attr twice fails.

fn main() {
    unsafe {
        use core::mem::MaybeUninit;
        let mut attr = MaybeUninit::<libc::pthread_attr_t>::uninit();

        libc::pthread_attr_init(attr.as_mut_ptr());

        libc::pthread_attr_destroy(attr.as_mut_ptr());

        libc::pthread_attr_destroy(attr.as_mut_ptr());
        //~^ ERROR: has not been initialized
    }
}
