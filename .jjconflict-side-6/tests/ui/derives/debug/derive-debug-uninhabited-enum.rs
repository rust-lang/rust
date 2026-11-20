//! Regression test for `#[derive(Debug)]` on enums with uninhabited variants.
//!
//! Ensures there are no special warnings about uninhabited types when deriving
//! Debug on an enum with uninhabited variants, only standard unused warnings.
//!
//! Issue: https://github.com/rust-lang/rust/issues/38885

//@ check-pass
//@ compile-flags: -Wunused

#[derive(Debug)]
enum Void {}

#[derive(Debug)]
enum Foo {
    Bar(#[allow(dead_code)] u8),
    Void(Void), //~ WARN variant `Void` is never constructed
}

fn main() {
    let x = Foo::Bar(42);
    println!("{:?}", x);
}
