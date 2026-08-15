// gate-test-const_async_blocks

//@ edition:2018
//@ revisions: with_feature without_feature
//@[with_feature] check-pass

#![cfg_attr(with_feature, feature(const_async_blocks))]

use std::future::Future;

// From <https://github.com/rust-lang/rust/issues/77361>
const _: i32 = { core::mem::ManuallyDrop::new(async { 0 }); 4 };
//[without_feature]~^ ERROR `async` block

static _FUT: &(dyn Future<Output = ()> + Sync) = &async {};
//[without_feature]~^ ERROR `async` block

fn main() {}
