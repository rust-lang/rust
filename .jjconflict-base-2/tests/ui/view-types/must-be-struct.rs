#![feature(view_types, view_type_macro)]
#![allow(unused)]

use std::view::view_type;

enum Foo {
    Bar,
    Baz,
}

fn f(_: view_type!(Foo.{})) {}
//~^ ERROR only structs can be viewed
fn g(_: view_type!(u8.{})) {}
//~^ ERROR only structs can be viewed
fn h(_: view_type!(char.{})) {}
//~^ ERROR only structs can be viewed

fn main() {}
