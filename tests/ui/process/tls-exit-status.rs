// run-fail
// error-pattern:nonzero
// exec-env:RUST_NEWRT=1
// ignore-emscripten no processes

use std::env;

fn main() {
    env::args();
    panic!("please have a nonzero exit status");
}
