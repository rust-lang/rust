//@ known-bug: #117808
//@ edition:2021
//@ needs-rustc-debug-assertions

use std::future::Future;

fn hrc<R, F: for<'a> AsyncClosure<'a, (), R>>(f: F) -> F {
    f
}

fn main() {
    hrc(|x| async {});
}

trait AsyncClosure<'a, I, R>
where
    I: 'a,
{
}

impl<'a, I, R, Fut, F> AsyncClosure<'a, I, R> for F
where
    I: 'a,
    F: Fn(&'a I) -> Fut,
    Fut: Future<Output = R> + Send + 'a,
{
}
