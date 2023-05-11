// run-pass

#![deny(dead_code)]

const GLOBAL_BAR: u32 = 1;

struct Foo;

impl Foo {
    const BAR: u32 = GLOBAL_BAR;
}

pub fn main() {
    let _: u32 = Foo::BAR;
}
