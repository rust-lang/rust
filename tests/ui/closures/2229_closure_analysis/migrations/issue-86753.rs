//@ edition:2018
//@ check-pass

#![warn(rust_2021_compatibility)]

use std::future::Future;

struct Runtime;

impl Runtime {
    pub fn block_on<F: Future>(&self, _future: F) -> F::Output {
        unimplemented!()
    }
}

pub fn http<F, Fut>(_func: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = ()>,
{
    let rt = Runtime {};
    let srv = rt.block_on(async move { serve(move || async move { unimplemented!() }) });
    let _ = || rt.block_on(async { srv });
}

pub struct Server<S> {
    _marker: std::marker::PhantomData<S>,
}

pub fn serve<S>(_new_service: S) -> Server<S> {
    unimplemented!()
}

fn main() { }
