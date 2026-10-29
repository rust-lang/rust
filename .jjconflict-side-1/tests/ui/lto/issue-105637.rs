// Regression test for issue #105637: `-Zdylib-lto` with LTO duplicated symbols from other dylibs,
// in this case from libstd.
//
// That manifested as both `rustc_driver` and rustc's "main" (`compiler/rustc`) having their own
// `std::panicking::HOOK` static, and the hook in rustc's main (the default stdlib's) being executed
// when rustc ICEs, instead of the overridden hook from `rustc_driver` (which also displays the
// query stack and information on how to open a GH issue for the encountered ICE).
//
// In this test, we reproduce this setup by installing a panic hook in both the main and an LTOed
// dylib: the last hook set should be the one being executed, the dylib's.

//@ aux-build: thinlto-dylib.rs
//@ run-fail
//@ check-run-results

extern crate thinlto_dylib;

use std::panic;

fn main() {
    // We don't want to see this panic hook executed
    std::panic::set_hook(Box::new(|_| {
        eprintln!("main crate panic hook");
    }));

    // Have the LTOed dylib install its own hook and panic, we want to see its hook executed.
    thinlto_dylib::main();
}
