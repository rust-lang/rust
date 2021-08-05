// error-pattern: the program aborted
#![feature(c_unwind)]

extern "C" fn panic_abort() { panic!() }

fn main() {
    panic_abort();
}
