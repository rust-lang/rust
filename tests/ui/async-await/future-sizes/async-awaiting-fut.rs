// revisions: both_off just_prop both_on
// ignore-tidy-linelength
// [both_off]  compile-flags: -Z print-type-sizes --crate-type lib -Z mir-enable-passes=-UpvarToLocalProp,-InlineFutureIntoFuture
// [just_prop] compile-flags: -Z print-type-sizes --crate-type lib -Z mir-enable-passes=+UpvarToLocalProp,-InlineFutureIntoFuture
// [both_on]   compile-flags: -Z print-type-sizes --crate-type lib -Z mir-enable-passes=+UpvarToLocalProp,+InlineFutureIntoFuture
// edition:2021
// build-pass
// ignore-pass

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
