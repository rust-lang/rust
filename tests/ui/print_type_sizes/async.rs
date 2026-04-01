// FIXME(#61117): Respect debuginfo-level-tests, do not force debuginfo=0
//@ compile-flags: -C debuginfo=0
//@ compile-flags: -C panic=abort -Z print-type-sizes --crate-type lib
//@ needs-deterministic-layouts
//@ edition:2021
//@ build-pass
//@ ignore-pass
//@ only-x86_64

#![allow(dropping_copy_types)]

async fn wait() {}

pub async fn test(arg: [u8; 8192]) {
    wait().await;
    drop(arg);
}
