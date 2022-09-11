// run-pass
#![allow(unreachable_code)]
#![deny(unused_variables)]

fn main() {
    for _ in match return () {
        () => Some(0),
    } {}
}
