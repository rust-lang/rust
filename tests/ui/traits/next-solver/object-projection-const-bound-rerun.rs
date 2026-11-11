//! Regression test for <https://github.com/rust-lang/rust/issues/159462>.
//@ compile-flags: -Znext-solver
//@ check-pass

trait Trait {
    type Assoc;
}
impl<T: Send> Trait for T
where
    [T; 1 + 1]: Sized,
{
    type Assoc = ();
}

trait Proj<T> {
    type Assoc;
}

trait Foo: Proj<<u8 as Trait>::Assoc, Assoc = ()> {
    fn m(&self);
}

fn f(x: &dyn Foo) {
    x.m()
}

fn main() {}
