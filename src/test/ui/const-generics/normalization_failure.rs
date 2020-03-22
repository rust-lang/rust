// run-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

fn main() {
    <()>::foo();
}

trait Foo<const X: usize> {
    fn foo() -> usize {
        X
    }
}

impl Foo<{3}> for () {}
