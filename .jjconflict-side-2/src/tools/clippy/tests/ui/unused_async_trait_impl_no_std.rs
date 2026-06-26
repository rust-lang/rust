#![no_std]
#![warn(clippy::unused_async_trait_impl)]

trait HasAsyncMethod {
    async fn do_something() -> u32;
}

struct Inefficient;

impl HasAsyncMethod for Inefficient {
    async fn do_something() -> u32 {
        //~^ unused_async_trait_impl
        1
    }
}
