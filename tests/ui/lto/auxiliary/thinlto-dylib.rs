// Auxiliary crate for test issue-105637: the LTOed dylib which had duplicate symbols from libstd,
// breaking the panic hook feature.
//
// This simulates the `rustc_driver` crate, and the main crate simulates rustc's main binary hooking
// into this driver.

//@ compile-flags: -Zdylib-lto -C lto=thin

use std::panic;

pub fn main() {
    // Install the hook we want to see executed
    panic::set_hook(Box::new(|_| {
        eprintln!("LTOed auxiliary crate panic hook");
    }));

    // Trigger the panic hook with an ICE
    run_compiler();
}

fn run_compiler() {
    panic!("ICEing");
}
