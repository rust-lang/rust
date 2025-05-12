//@ignore-target: windows # No pthreads on Windows
//@ignore-target: apple # Our macOS condattr don't have any fields so we do not notice this.

/// Test that destroying a pthread_condattr twice fails, even without a check for number validity

fn main() {
    unsafe {
        use core::mem::MaybeUninit;
        let mut attr = MaybeUninit::<libc::pthread_condattr_t>::uninit();

        libc::pthread_condattr_init(attr.as_mut_ptr());

        libc::pthread_condattr_destroy(attr.as_mut_ptr());

        libc::pthread_condattr_destroy(attr.as_mut_ptr());
        //~^ ERROR: Undefined Behavior: using uninitialized data, but this operation requires initialized memory
    }
}
