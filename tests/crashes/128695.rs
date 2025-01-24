//@ known-bug: rust-lang/rust#128695
//@ edition: 2021

use core::pin::{pin, Pin};

fn main() {
    let fut = pin!(async {
        let async_drop_fut = pin!(core::future::async_drop(async {}));
        (async_drop_fut).await;
    });
}
