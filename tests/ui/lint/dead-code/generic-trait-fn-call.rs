//@ check-pass

#![deny(dead_code)]

trait Trait {
    fn method(&self);
}

struct S;

impl Trait for S {
    fn method(&self) {
        helper();
    }
}

fn helper() {}

fn invoke<T: Trait>(x: T) {
    x.method();
}

#[allow(dead_code)]
fn unused() {
    invoke(S);
}

fn main() {}
