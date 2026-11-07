//@ check-pass

#![deny(dead_code)]

#[allow(dead_code)]
trait Trait {
    fn method();
}

#[allow(dead_code)]
struct Struct;

impl Trait for Struct {
    fn method() {
        helper();
    }
}

fn helper() {}

fn main() {}
