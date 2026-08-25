//@ edition: 2024
// When pub async fn is monomorphized, its implementation coroutine is also monomorphized
//@ compile-flags: --crate-type=lib

//~ MONO_ITEM fn async_fn @@
//~ MONO_ITEM fn async_fn::{closure#0} @@
#[unsafe(no_mangle)]
pub async fn async_fn(x: u64) -> bool {
    true
}
