// revisions: both_off just_prop both_on
// ignore-tidy-linelength
// [both_off]  compile-flags: -Z print-type-sizes --crate-type lib -Z mir-enable-passes=-UpvarToLocalProp,-InlineFutureIntoFuture
// [just_prop] compile-flags: -Z print-type-sizes --crate-type lib -Z mir-enable-passes=+UpvarToLocalProp,-InlineFutureIntoFuture
// [both_on]   compile-flags: -Z print-type-sizes --crate-type lib -Z mir-enable-passes=+UpvarToLocalProp,+InlineFutureIntoFuture
// edition:2021
// build-pass
// ignore-pass

#![allow(dropping_copy_types)]

async fn wait() {}

pub async fn test(arg: [u8; 8192]) {
    wait().await;
    drop(arg);
}
