//@ check-fail

#![feature(exhaustive_patterns, never_type)]

mod inner {
    pub struct Wrapper<T>(T);
}

enum Either<A, B> {
    A(A),
    B(inner::Wrapper<B>),
}

fn foo() -> Either<(), !> {
    Either::A(())
}

fn main() {
    // We can't treat this a irrefutable, because `Either::B` could become
    // inhabited in the future because it's private.
    let Either::A(()) = foo();
    //~^ error refutable pattern in local binding
}
