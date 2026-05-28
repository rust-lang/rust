//! regression test for <https://github.com/rust-lang/rust/issues/11680>
//@ aux-build:imported-enum-is-private.rs

extern crate imported_enum_is_private as other;

fn main() {
    let _b = other::Foo::Bar(1);
    //~^ ERROR: enum `Foo` is private

    let _b = other::test::Foo::Bar(1);
    //~^ ERROR: enum `Foo` is private
}
