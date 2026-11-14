//@ edition:2024

#![feature(async_drop)]
#![allow(incomplete_features)]
#![allow(long_running_const_eval)]

use core::future::AsyncDrop;
use core::pin::Pin;

struct AsyncDtor;

impl Drop for AsyncDtor {
    fn drop(&mut self) {}
}

impl AsyncDrop for AsyncDtor {
    async fn drop(self: Pin<&mut Self>) {
        core::future::ready(()).await;
    }
}

pub async fn has_const() {
    const CONST: () = {
        let mut i = 0;

        while i < 10_000_000 {
            i += 1;
        }
    };

    let _guard = AsyncDtor;

    core::future::ready(()).await;

    CONST
}
