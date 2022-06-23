#![feature(c_unwind)]

extern "C" fn panic_abort() {
    //~^ ERROR: the program aborted
    panic!()
}

fn main() {
    panic_abort();
}
