//@ aux-build:other.rs
//@ reference: attributes.diagnostic.on_unimplemented.intro

extern crate other;

use other::Foo;

fn take_foo(_: impl Foo) {}

fn main() {
    take_foo(());
    //~^ERROR Message
}
