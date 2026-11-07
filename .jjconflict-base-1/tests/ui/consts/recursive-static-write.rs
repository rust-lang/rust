//! Ensure that writing to `S` while initializing `S` errors.
//! Regression test for <https://github.com/rust-lang/rust/issues/142404>.
#![allow(dead_code)]

struct Foo {
    x: i32,
    y: (),
}

static S: Foo = Foo {
    x: 0,
    y: unsafe {
        (&raw const S.x).cast_mut().write(1); //~ERROR access itself during initialization
    },
};

static mut S2: Foo = Foo {
    x: 0,
    y: unsafe {
        S2.x = 1; //~ERROR access itself during initialization
    },
};

fn main() {}
