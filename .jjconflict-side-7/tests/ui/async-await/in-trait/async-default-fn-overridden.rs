//@ run-pass
//@ edition:2021

use std::future::Future;

trait AsyncTrait {
    #[allow(async_fn_in_trait)]
    async fn default_impl() {
        assert!(false);
    }

    #[allow(async_fn_in_trait)]
    async fn call_default_impl() {
        Self::default_impl().await
    }
}

struct AsyncType;

impl AsyncTrait for AsyncType {
    async fn default_impl() {
        // :)
    }
}

async fn async_main() {
    // Should not assert false
    AsyncType::call_default_impl().await;
}

// ------------------------------------------------------------------------- //
// Implementation Details Below...

use std::pin::pin;
use std::task::*;

fn main() {
    let mut fut = pin!(async_main());

    // Poll loop, just to test the future...
    let ctx = &mut Context::from_waker(Waker::noop());

    loop {
        match fut.as_mut().poll(ctx) {
            Poll::Pending => {}
            Poll::Ready(()) => break,
        }
    }
}
