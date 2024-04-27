//@ run-fail
//@ error-pattern:nonzero
//@ exec-env:RUST_NEWRT=1
//@ ignore-wasm32 no processes

use std::env;

fn main() {
    env::args();
    panic!("please have a nonzero exit status");
}
