// run-fail
//@error-in-other-file:nonzero
// exec-env:RUST_NEWRT=1
//@ignore-target-emscripten no processes

use std::env;

fn main() {
    env::args();
    panic!("please have a nonzero exit status");
}
