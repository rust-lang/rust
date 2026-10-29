//@ run-pass
//@ check-run-results
//@ only-gnu
//@ only-linux
//
// Regression test for #130210.
// .init_array doesn't work everywhere, so we limit the test to just GNU/Linux.

use std::ffi::c_int;
use std::thread;

#[used]
#[link_section = ".init_array"]
static INIT: extern "C" fn(c_int, *const *const u8, *const *const u8) = {
    extern "C" fn init(_argc: c_int, _argv: *const *const u8, _envp: *const *const u8) {
        print!("Hello from before ");
    }

    init
};

fn main() {
    println!("{}!", thread::current().name().unwrap());
}
