#![feature(rustc_attrs)]
#![allow(warnings)]

#[derive(Debug)]
struct Point {
}

struct NestedA<'a, 'b> {
    x: &'a NestedB<'b>
    //~^ ERROR E0491
}

struct NestedB<'a> {
    x: &'a i32,
}

fn main() {
}
