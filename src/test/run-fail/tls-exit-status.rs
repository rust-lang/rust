// error-pattern:nonzero
// exec-env:RUST_NEWRT=1
// ignore-cloudabi no std::env

use std::env;

fn main() {
    env::args();
    panic!("please have a nonzero exit status");
}
