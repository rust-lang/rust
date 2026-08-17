//@ edition: 2021
//@ compile-flags: -Clink-dead-code --crate-type=lib -Copt-level=0

//~ MONO_ITEM fn async_fn @@
//~ MONO_ITEM fn async_fn::{closure#0} @@
//~ MONO_ITEM fn std::ops::coroutine::adapters::future_from_coroutine::<{static coroutine body of async_fn()}> @@
pub async fn async_fn() {}

//~ MONO_ITEM fn closure @@
//~ MONO_ITEM fn closure::{closure#0} @@
pub fn closure() {
    let _ = || {};
}

//~ MONO_ITEM fn A::{constant#0}::{closure#0} @@
trait A where
    [(); (|| {}, 1).1]: Sized,
{
}
