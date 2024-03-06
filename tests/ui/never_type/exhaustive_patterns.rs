//@ check-fail
//@ known-bug: #104034

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
    let Either::A(()) = foo();
}
