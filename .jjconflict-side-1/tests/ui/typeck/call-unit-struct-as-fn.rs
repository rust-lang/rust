//! Regression test for <https://github.com/rust-lang/rust/issues/46771>.
//! Test calling unit struct as fn doesn't ICE.

fn main() {
    struct Foo;
    (1 .. 2).find(|_| Foo(0) == 0); //~ ERROR expected function, found `Foo`
}
