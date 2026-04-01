//@ check-pass
#![allow(unused_variables)]

trait Trait<Input> {
    type Output;

    fn method(&self, i: Input) -> bool { false }
}

fn main() {}
