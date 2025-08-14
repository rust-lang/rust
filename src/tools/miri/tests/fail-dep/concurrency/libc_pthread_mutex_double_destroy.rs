//@ignore-target: windows # No pthreads on Windows
//@ normalize-stderr-test: "(\n)ALLOC \(.*\) \{\n(.*\n)*\}(\n)" -> "${1}ALLOC DUMP${3}"
//@ normalize-stderr-test: "\[0x[0-9a-z]..0x[0-9a-z]\]" -> "[0xX..0xY]"

/// Test that destroying a pthread_mutex twice fails, even without a check for number validity

fn main() {
    unsafe {
        use core::mem::MaybeUninit;

        let mut attr = MaybeUninit::<libc::pthread_mutexattr_t>::uninit();
        libc::pthread_mutexattr_init(attr.as_mut_ptr());

        let mut mutex = MaybeUninit::<libc::pthread_mutex_t>::uninit();

        libc::pthread_mutex_init(mutex.as_mut_ptr(), attr.as_ptr());

        libc::pthread_mutex_destroy(mutex.as_mut_ptr());

        libc::pthread_mutex_destroy(mutex.as_mut_ptr());
        //~^ ERROR: /Undefined Behavior: reading memory .*, but memory is uninitialized/
    }
}
