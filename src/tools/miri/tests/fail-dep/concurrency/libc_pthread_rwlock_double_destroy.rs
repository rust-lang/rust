//@ignore-target: windows # No pthreads on Windows
//@ normalize-stderr-test: "(\n)ALLOC \(.*\) \{\n(.*\n)*\}(\n)" -> "${1}ALLOC DUMP${3}"
//@ normalize-stderr-test: "\[0x[0-9a-z]..0x[0-9a-z]\]" -> "[0xX..0xY]"

/// Test that destroying a pthread_rwlock twice fails, even without a check for number validity

fn main() {
    unsafe {
        let mut lock = libc::PTHREAD_RWLOCK_INITIALIZER;

        libc::pthread_rwlock_destroy(&mut lock);

        libc::pthread_rwlock_destroy(&mut lock);
        //~^ ERROR: /Undefined Behavior: reading memory .*, but memory is uninitialized/
    }
}
