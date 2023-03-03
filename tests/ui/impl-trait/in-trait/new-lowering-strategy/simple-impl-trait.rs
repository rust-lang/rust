// check-pass
// compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty

#![feature(return_position_impl_trait_in_trait)]
#![allow(incomplete_features)]

trait Foo {
    fn foo() -> impl Sized;
}

impl Foo for String {
    fn foo() -> i32 {
        22
    }
}

fn main() {}
