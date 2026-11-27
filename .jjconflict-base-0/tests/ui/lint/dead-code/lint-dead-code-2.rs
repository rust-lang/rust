#![allow(unused_variables)]
#![deny(dead_code)]
#![feature(rustc_attrs)]

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

fn used_fn() {}

#[rustc_main]
fn actual_main() {
    used_fn();
    let foo = Foo;
    foo.bar2();
}

// this is not main
fn main() { //~ ERROR: function `main` is never used
    dead_fn();
}
