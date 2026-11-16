//@ dont-require-annotations: ERROR
//@ compile-flags: --crate-type lib -Z ui-testing=no

#![feature(rustc_attrs)]

#[rustc_comptime]
fn comptime() {}

fn from_runtime() {
    comptime(); // ERROR

    const { comptime() }; // OK
}

// comptime functions cannot be called from const
// functions, as these may be called at runtime
const fn const_or_runtime() {
    comptime();
}
