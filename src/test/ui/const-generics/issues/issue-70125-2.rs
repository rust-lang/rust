// run-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

fn main() {
    <()>::foo();
}

trait Foo<const X: usize> {
    fn foo() -> usize {
        X
    }
}

impl Foo<3> for () {}
