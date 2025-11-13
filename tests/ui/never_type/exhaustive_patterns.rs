// Check that we don't consider types which aren't publicly uninhabited as
// uninhabited for purposes of pattern matching.
//
//@ check-fail

#![feature(never_type)]

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
