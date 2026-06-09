// Regression test for #79902

// Check that evaluation (which is used to determine whether to copy a type in
// MIR building) evaluates bounds from normalizing an impl after evaluating
// any bounds on the impl.

//@ check-pass

#![allow(dropping_copy_types)]

trait A {
    type B;
}
trait M {}

struct G<T, U>(*const T, *const U);

impl<T, U> Clone for G<T, U> {
    fn clone(&self) -> Self {
        G { ..*self }
    }
}

impl<T, U> Copy for G<T, U::B>
where
    T: A<B = U>,
    U: A,
{
}

impl A for () {
    type B = ();
}

fn is_m<T: M>(_: T) {}

fn main() {
    let x = G(&(), &());
    drop(x);
    drop(x);
}
