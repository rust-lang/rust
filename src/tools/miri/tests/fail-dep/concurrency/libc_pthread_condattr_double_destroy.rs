//@ignore-target: windows # No pthreads on Windows
//@ignore-target: apple # Our macOS condattr don't have any fields so we do not notice this.
//@ normalize-stderr-test: "(\n)ALLOC \(.*\) \{\n(.*\n)*\}(\n)" -> "${1}ALLOC DUMP${3}"
//@ normalize-stderr-test: "\[0x[0-9a-z]..0x[0-9a-z]\]" -> "[0xX..0xY]"

/// Test that destroying a pthread_condattr twice fails, even without a check for number validity

fn main() {
    unsafe {
        use core::mem::MaybeUninit;
        let mut attr = MaybeUninit::<libc::pthread_condattr_t>::uninit();

        libc::pthread_condattr_init(attr.as_mut_ptr());

        libc::pthread_condattr_destroy(attr.as_mut_ptr());

        libc::pthread_condattr_destroy(attr.as_mut_ptr());
        //~^ ERROR: /Undefined Behavior: reading memory .*, but memory is uninitialized/
    }
}
