//@ignore-target: windows # no libc
//@ revisions: default null
//@[null] compile-flags: -Zmiri-mute-stdout-stderr

fn main() {
    // This is std library UB, but that's not relevant since we're
    // only interacting with libc here.
    unsafe {
        libc::close(0);
        libc::close(1);
        libc::close(2);
    }
}
