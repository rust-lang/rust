// build-pass (FIXME(62277): could be check-pass?)

#![feature(rustc_attrs)]
#![allow(warnings)]

#[derive(Debug)]
struct Point {
}

struct NestedA<'a, 'b> {
    x: &'a NestedB<'b>
}

struct NestedB<'a> {
    x: &'a i32,
}

fn main() {
}
