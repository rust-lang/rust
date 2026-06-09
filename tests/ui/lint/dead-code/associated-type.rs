//@ run-pass

#![deny(dead_code)]

trait Foo {
    type Bar;
}

struct Used;

struct Ex;

impl Foo for Ex {
    type Bar = Used;
}

pub fn main() {
    let _x: &dyn Foo<Bar = <Ex as Foo>::Bar> = &Ex;
}
