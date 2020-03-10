#![feature(never_type, rustc_attrs)]
#![crate_type = "lib"]

enum E { Foo, Bar(!, i32, i32) }

#[rustc_layout(debug)]
type Test = E; //~ ERROR: layout debugging
