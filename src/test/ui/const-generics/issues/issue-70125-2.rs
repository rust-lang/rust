// run-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

fn main() {
    <()>::foo();
}

trait Foo<const X: usize> {
    fn foo() -> usize {
        X
    }
}

impl Foo<3> for () {}
