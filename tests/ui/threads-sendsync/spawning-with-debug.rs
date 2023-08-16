//@run
#![allow(unused_must_use)]
#![allow(unused_mut)]
//@ignore-target-windows
// exec-env:RUST_LOG=debug
//@ignore-target-emscripten no threads support

// regression test for issue #10405, make sure we don't call println! too soon.

use std::thread::Builder;

pub fn main() {
    let mut t = Builder::new();
    t.spawn(move || ());
}
