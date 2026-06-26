#![feature(view_types, view_type_macro)]
#![allow(unused)]

use std::view::view_type;

struct S {
    foo: (),
}

fn f(_: view_type!(S.{ bar })) {}
//~^ ERROR no field `bar` on type `S`
fn g(_: view_type!(S.{ foo, bar })) {}
//~^ ERROR no field `bar` on type `S`

fn main() {}
