//! Regression test for <https://github.com/rust-lang/rust/issues/18173>.

//@ check-pass
trait Foo {
    type T;
}

// should be able to use a trait with an associated type without specifying it as an argument
trait Bar<F: Foo> {
    fn bar(foo: &F);
}

pub fn main() {
}
