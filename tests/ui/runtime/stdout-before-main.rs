//@ run-pass
//@ check-run-results
//@ only-gnu
//@ only-linux

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
    print!("{}!", thread::current().name().unwrap());
}
