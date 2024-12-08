// gate-test-const_async_blocks

//@ edition:2018
//@ revisions: with_feature without_feature

#![feature(rustc_attrs)]
#![cfg_attr(with_feature, feature(const_async_blocks))]

use std::future::Future;

// From <https://github.com/rust-lang/rust/issues/77361>
const _: i32 = { core::mem::ManuallyDrop::new(async { 0 }); 4 };
//[without_feature]~^ `async` block

static _FUT: &(dyn Future<Output = ()> + Sync) = &async {};
//[without_feature]~^ `async` block

#[rustc_error]
fn main() {} //[with_feature]~ fatal error triggered by #[rustc_error]
