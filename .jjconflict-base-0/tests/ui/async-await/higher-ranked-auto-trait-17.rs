// Repro for <https://github.com/rust-lang/rust/issues/114177#issue-1826550174>.
//@ edition: 2021
//@ revisions: assumptions no_assumptions
//@[assumptions] compile-flags: -Zhigher-ranked-assumptions
//@[assumptions] check-pass
//@[no_assumptions] known-bug: #110338

// Using `impl Future` instead of `async to ensure that the Future is Send.
//
// In the original code `a` would be `&[T]`. For more minimization I've removed the reference.
fn foo(a: [(); 0]) -> impl std::future::Future<Output = ()> + Send {
    async move {
        let iter = Adaptor::new(a.iter().map(|_: &()| {}));
        std::future::pending::<()>().await;
        drop(iter);
    }
}

struct Adaptor<T: Iterator> {
    iter: T,
    v: T::Item,
}

impl<T: Iterator> Adaptor<T> {
    pub fn new(_: T) -> Self {
        Self { iter: todo!(), v: todo!() }
    }
}

fn main() {}
