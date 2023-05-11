#![feature(c_unwind)]

extern "C" fn panic_abort() {
    //~^ ERROR: panic in a function that cannot unwind
    panic!()
}

fn main() {
    panic_abort();
}
