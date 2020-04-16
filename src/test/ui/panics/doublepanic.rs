#![allow(unreachable_code)]

// run-fail
// error-pattern:One

fn main() {
    panic!("One");
    panic!("Two");
}
