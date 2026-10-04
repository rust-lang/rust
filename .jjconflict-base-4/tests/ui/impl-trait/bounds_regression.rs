//@ run-pass

pub trait FakeCoroutine {
    type Yield;
    type Return;
}

pub trait FakeFuture {
    type Output;
}

pub fn future_from_coroutine<
    T: FakeCoroutine<Yield = ()>
>(x: T) -> impl FakeFuture<Output = T::Return> {
    GenFuture(x)
}

struct GenFuture<T: FakeCoroutine<Yield = ()>>(#[allow(dead_code)] T);

impl<T: FakeCoroutine<Yield = ()>> FakeFuture for GenFuture<T> {
    type Output = T::Return;
}

fn main() {}
