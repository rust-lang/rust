//@ run-pass
#![allow(unused_must_use)]
#![allow(unused_mut)]
//@ exec-env:RUST_LOG=debug
//@ needs-threads

// regression test for issue #10405, make sure we don't call println! too soon.

use std::thread::Builder;

pub fn main() {
    let mut t = Builder::new();
    t.spawn(move || ());
}
