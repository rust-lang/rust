// Repro for <https://github.com/rust-lang/rust/issues/130113#issue-2512517191>.
//@ edition: 2021
//@ revisions: assumptions no_assumptions
//@[assumptions] compile-flags: -Zhigher-ranked-assumptions
//@[assumptions] check-pass
//@[no_assumptions] known-bug: #110338

use std::future::Future;

fn main() {
    let call_me = Wrap(CallMeImpl { value: "test" });

    assert_send(async {
        call_me.call().await;
    });
}

pub fn assert_send<F>(_future: F)
where
    F: Future + Send,
{
}

pub trait CallMe {
    fn call(&self) -> impl Future<Output = ()> + Send;
}

struct Wrap<T>(T);

impl<S> CallMe for Wrap<S>
where
    S: CallMe + Send,
{
    // adding `+ Send` to this RPIT fixes the issue
    fn call(&self) -> impl Future<Output = ()> {
        self.0.call()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CallMeImpl<T> {
    value: T,
}

impl<T> CallMe for CallMeImpl<T>
where
    // Can replace `Send` by `ToString`, `Clone`, whatever. When removing the
    // `Send` bound, the compiler produces a higher-ranked lifetime error.
    T: Send + 'static,
{
    fn call(&self) -> impl Future<Output = ()> {
        async {}
    }
}
