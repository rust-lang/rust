#![allow(dead_code)]

use std::cell::Cell;

struct Foo<'a: 'b, 'b> {
    x: Cell<&'a u32>,
    y: Cell<&'b u32>,
}

fn bar<'a, 'b>(x: Cell<&'a u32>, y: Cell<&'b u32>) {
    Foo { x, y };
    //~^ ERROR lifetime may not live long enough
}

fn main() {}
