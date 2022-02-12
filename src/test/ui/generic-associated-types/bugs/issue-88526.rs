// check-fail

// This should pass, but requires more logic.

#![feature(generic_associated_types)]

trait A {
    type I<'a>;
}

pub struct TestA<F>
{
    f: F,
}

impl<F> A for TestA<F> {
    type I<'a> = &'a F;
}

struct TestB<Q, F>
{
    q: Q,
    f: F,
}

impl<'q, Q, I, F> A for TestB<Q, F> //~ the type parameter
where
    Q: A<I<'q> = &'q I>,
    F: Fn(I),
{
    type I<'a> = ();
}

fn main() {}
