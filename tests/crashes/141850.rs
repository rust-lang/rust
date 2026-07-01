//@ known-bug: rust-lang/rust#141850
//@edition: 2024
//@ compile-flags: -Copt-level=0
#![feature(pin_ergonomics)]
async fn a() {
    wrapper_call(handler).await;
}
async fn wrapper_call<F>(_: F) -> F::Output
where
    F: Handler,
{
    todo!()
}
async fn handler();
trait Handler {
    type Output;
}
impl<Func, Fut> Handler for Func
where
    Func: Fn() -> Fut,
    Fut: Future,
{
    type Output = Fut;
}
