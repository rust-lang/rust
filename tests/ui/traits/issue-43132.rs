//@ run-pass
#![allow(unused)]

fn main() {
}

fn foo() {
    let b = mk::<
        Forward<(Box<dyn Future<Error = u32>>,)>,
    >();
    b.map_err(|_| ()).join();
}

fn mk<T>() -> T {
    loop {}
}

impl<I: Future<Error = E>, E> Future for (I,) {
    type Error = E;
}

struct Forward<T: Future> {
    _a: T,
}

impl<T: Future> Future for Forward<T>
where
    T::Error: From<u32>,
{
    type Error = T::Error;
}

trait Future {
    type Error;

    fn map_err<F, E>(self, _: F) -> (Self, F)
    where
        F: FnOnce(Self::Error) -> E,
        Self: Sized,
    {
        loop {}
    }

    fn join(self) -> (MaybeDone<Self>, ())
    where
        Self: Sized,
    {
        loop {}
    }
}

impl<S: ?Sized + Future> Future for Box<S> {
    type Error = S::Error;
}

enum MaybeDone<A: Future> {
    _Done(A::Error),
}

impl<U, A: Future, F> Future for (A, F)
where
    F: FnOnce(A::Error) -> U,
{
    type Error = U;
}
