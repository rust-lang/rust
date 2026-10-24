// ignore-tidy-linelength
//@ edition: 2024
//@[with_feature] filecheck-flags: --input-file build/host/test/mir-opt/coroutine/async_drop_shim_usage.with_feature/core.future-async_drop-async_drop_in_place-{closure#0}.HasHasDrop.coroutine_before.0.mir
//@[without_feature] filecheck-flags: --input-file build/host/test/mir-opt/coroutine/async_drop_shim_usage.without_feature/core.future-async_drop-async_drop_in_place-{closure#0}.HasHasDrop.coroutine_before.0.mir
//@ revisions: with_feature without_feature
//@ aux-build: has_drop_inside_dep.rs

// WARNING: If you would ever want to modify this test,
// please consider modifying rustc's async drop test at
// `tests/ui/async-await/async-drop/async-drop-shim-usage.rs`.

#![cfg_attr(with_feature, feature(async_drop))]
#![allow(incomplete_features)]

extern crate has_drop_inside_dep;

use has_drop_inside_dep::{with_has_drop, with_has_has_drop};

// with_feature-LABEL: fn async_drop_in_place::{closure#0}(_1: {async fn body of async_drop_in_place<HasHasDrop>()}, _2: std::future::ResumeTy) -> ()
// with_feature: [[f:_.*]] = async_drop_in_place::<HasDrop>(

// without_feature-LABEL: fn async_drop_in_place::{closure#0}(_1: {async fn body of async_drop_in_place<HasHasDrop>()}, _2: std::future::ResumeTy) -> ()
// without_feature: [[f:_.*]] = async_drop_in_place::<HasDrop>(

#[allow(unused)]
async fn usage_with_has_drop() {
    let _f = with_has_drop();
}

#[allow(unused)]
async fn usage_with_has_has_drop() {
    let _f = with_has_has_drop();
}
