// compile-flags: -Z print-type-sizes
// edition:2021
// build-pass
// ignore-pass

#![feature(start)]

async fn wait() {}

async fn test(arg: [u8; 8192]) {
    wait().await;
    drop(arg);
}

#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    let _ = test([0; 8192]);
    0
}
