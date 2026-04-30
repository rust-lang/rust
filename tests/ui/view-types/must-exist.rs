#![feature(view_types, view_type_macro)]
//~^ ERROR unknown feature `view_type_macro`
#![allow(unused)]

use std::view::view_type;
//~^ ERROR unresolved import `std::view`

struct S {
    foo: (),
}

// We expect  errors here, since `S` has no field `bar`.
fn f(_: view_type!(S.{ bar })) {}
fn g(_: view_type!(S.{ foo, bar })) {}

fn main() {}
