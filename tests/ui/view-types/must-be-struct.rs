//@ known-bug: unknown
//@ run-pass

#![feature(view_types, view_type_macro)]
#![allow(unused)]

use std::view::view_type;

enum Foo {
    Bar,
    Baz,
}

// The following types are not structs, we expect errors here.
fn f(_: view_type!(Foo.{})) {}
fn g(_: view_type!(u8.{})) {}
fn h(_: view_type!(char.{})) {}

fn main() {}
