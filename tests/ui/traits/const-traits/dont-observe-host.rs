//@ compile-flags: -Znext-solver
//@ check-pass

#![feature(const_trait_impl)]

#[const_trait]
trait Trait {
    fn method() {}
}

impl const Trait for () {}

fn main() {
    let mut x = const {
        let x = <()>::method;
        x();
        x
    };
    let y = <()>::method;
    y();
    x = y;
}
