//! Regression test for <https://github.com/rust-lang/rust/issues/162391>.
#![feature(gca_min_const_items, gca_macroless_items)]

enum Foo {
    A { y: NOPE },
    //~^ ERROR cannot find type `NOPE` in this scope
}

const LIST: [Foo; 0] = [];

fn main() {
    match LIST {
        _ => {}
    }
}
