//! Regression test for <https://github.com/rust-lang/rust/issues/42210>.
//! Tuple with unsized tail caused ICE.

//@ run-pass
//@ compile-flags: -g

trait Foo {
    fn foo() { }
}

struct Bar;

trait Baz {
}

impl Foo for (Bar, dyn Baz) { }


fn main() {
    <(Bar, dyn Baz) as Foo>::foo()
}
