//@ compile-flags: -Znext-solver
//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/97487>.
// Used to ask for type annotations
trait A {
    type X;
}

trait B: for<'a> A<X = u32> {
}

impl<T> A for T where T: B {
    type X = u32;
}

fn main() {}
