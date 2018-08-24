#![feature(rustc_attrs)]
#![allow(warnings)]

// This used to ICE because the "if" being unreachable was not handled correctly
fn err() {
    if loop {} {}
}

#[rustc_error]
fn main() {} //~ ERROR compilation successful
