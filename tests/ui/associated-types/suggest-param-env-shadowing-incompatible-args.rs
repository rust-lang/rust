//@ compile-flags: -Znext-solver=globally

// Regression test for https://github.com/rust-lang/rust/issues/152684.

#![feature(associated_type_defaults)]

trait Foo {
    type Assoc<T = u8> = T;
    //~^ ERROR defaults for generic parameters are not allowed here
    fn foo() -> Self::Assoc;
}
impl Foo for () {
    fn foo() -> Self::Assoc {
        [] //~ ERROR mismatched types
    }
}

fn main() {}
