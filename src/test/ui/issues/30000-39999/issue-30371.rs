// run-pass
#![allow(unreachable_code)]
#![allow(unused_mut)] // rust-lang/rust#54586
#![deny(unused_variables)]

fn main() {
    for _ in match return () {
        () => Some(0),
    } {}
}
