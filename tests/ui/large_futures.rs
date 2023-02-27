// run-rustfix

#![feature(generators)]
#![warn(clippy::large_futures)]
#![allow(clippy::future_not_send)]
#![allow(clippy::manual_async_fn)]

async fn big_fut(_arg: [u8; 1024 * 16]) {}

async fn wait() {
    let f = async {
        big_fut([0u8; 1024 * 16]).await;
    };
    f.await
}
async fn calls_fut(fut: impl std::future::Future<Output = ()>) {
    loop {
        wait().await;
        if true {
            return fut.await;
        } else {
            wait().await;
        }
    }
}

pub async fn test() {
    let fut = big_fut([0u8; 1024 * 16]);
    foo().await;
    calls_fut(fut).await;
}

pub fn foo() -> impl std::future::Future<Output = ()> {
    async {
        let x = [0i32; 1024 * 16];
        async {}.await;
        dbg!(x);
    }
}

fn main() {}
