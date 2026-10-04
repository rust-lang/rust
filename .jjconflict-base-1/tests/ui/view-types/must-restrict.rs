//@ known-bug: unknown
//@ run-pass

#![feature(view_types, view_type_macro)]
#![allow(unused)]

use std::view::view_type;

struct S {
    foo: (),
    bar: (),
}

// The outermost fields are supersets of the innermost views, we expect this to trigger an error.
fn f(_: view_type!(view_type!(S.{}).{ foo })) {}
fn g(_: view_type!(view_type!(S.{ foo }).{ bar })) {}
fn h(_: view_type!(view_type!(view_type!(S.{ foo }).{}).{ foo })) {}

fn main() {}
