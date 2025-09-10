// FIXME(#61117): Respect debuginfo-level-tests, do not force debuginfo=0
//@ compile-flags: -C debuginfo=0
//@ compile-flags: -C panic=abort -Z print-type-sizes --crate-type lib
//@ needs-deterministic-layouts
//@ edition:2021
//@ build-pass
//@ ignore-pass
//@ only-x86_64

async fn wait() {}

async fn big_fut(arg: [u8; 1024]) {}

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
    let fut = big_fut([0u8; 1024]);
    calls_fut(fut).await;
}
