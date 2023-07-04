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
    let Either::A(()) = foo();
    //~^ ERROR refutable pattern in local binding
}
