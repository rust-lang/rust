run-fail
#![allow(unreachable_code)]

// error-pattern:One
fn main() {
    panic!("One");
    panic!("Two");
}
