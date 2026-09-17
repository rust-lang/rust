//@ edition: 2024

use std::future::Future;

fn unit_return_and_async_block_type_mismatch() -> impl Future<Output = ()> {
    //~^ ERROR: `()` is not a future [E0277]
    if false {
        return;
    }
    async {}
    //~^ ERROR: mismatched types [E0308]
}

// should produce boxing suggestion
fn multiple_returns_same_future_output() -> impl Future<Output = ()> {
    if false {
        return async {};
    }
    async {}
    //~^ ERROR: mismatched types [E0308]
}

fn main() {}
