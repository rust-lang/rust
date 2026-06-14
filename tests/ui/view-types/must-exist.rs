//@ known-bug: unknown
//@ check-pass

#![feature(view_types)]
#![allow(unused)]

struct S {
    foo: (),
}

fn f(_: S.{ bar }) {}
fn g(_: S.{ foo, bar }) {}

fn main() {}
