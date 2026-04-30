//@ known-bug: unknown

#![feature(view_types)]
#![allow(unused)]

enum Foo {
    Bar,
    Baz,
}

struct Cat  {
    mrow: (),
    mrrp: (),
}

fn f(_: Foo.{}) {}
fn g(_: u8.{}) {}
fn h(_: char.{}) {}

fn i(_: Cat.{ mrow, mrrp }) {}

fn main() {}
