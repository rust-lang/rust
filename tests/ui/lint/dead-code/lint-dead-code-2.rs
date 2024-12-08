#![allow(unused_variables)]
#![deny(dead_code)]
#![feature(rustc_attrs, start)]

struct Foo;

trait Bar {
    fn bar1(&self);
    fn bar2(&self) {
        self.bar1();
    }
}

impl Bar for Foo {
    fn bar1(&self) {
        live_fn();
    }
}

fn live_fn() {}

fn dead_fn() {} //~ ERROR: function `dead_fn` is never used

#[rustc_main]
fn dead_fn2() {} //~ ERROR: function `dead_fn2` is never used

fn used_fn() {}

#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    used_fn();
    let foo = Foo;
    foo.bar2();
    0
}

// this is not main
fn main() { //~ ERROR: function `main` is never used
    dead_fn();
    dead_fn2();
}
