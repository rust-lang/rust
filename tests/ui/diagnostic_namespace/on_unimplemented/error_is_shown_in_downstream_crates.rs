//@ aux-build:other.rs

extern crate other;

use other::Foo;

fn take_foo(_: impl Foo) {}

fn main() {
    take_foo(());
    //~^ERROR Message
}
