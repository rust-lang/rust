//@ compile-flags: -Z print-type-sizes --crate-type lib
//@ edition:2021
//@ build-pass
//@ ignore-pass

#![allow(dropping_copy_types)]

async fn wait() {}

pub async fn test(arg: [u8; 8192]) {
    wait().await;
    drop(arg);
}
