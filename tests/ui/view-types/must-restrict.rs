//@ known-bug: unknown

#![feature(view_types)]
#![allow(unused)]

struct S {
    foo: (),
    bar: (),
}

fn f(_: S.{}.{ foo }) {}
fn g(_: S.{ foo }.{ bar }) {}
fn h(_: S.{ foo }.{}.{ foo }) {}

fn main() {}
