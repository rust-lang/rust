//@ edition: 2021
//@ compile-flags: -Clink-dead-code --crate-type=lib

//~ MONO_ITEM fn async_fn @@
//~ MONO_ITEM fn async_fn::{closure#0} @@
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
