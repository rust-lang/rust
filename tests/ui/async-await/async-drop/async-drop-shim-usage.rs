//@ edition: 2024
//@ run-pass
//@ check-run-results
//@ revisions: with_feature without_feature
//@ aux-build: has-drop-inside-dep.rs

// WARNING: If you would ever want to modify this test,
// please consider modifying rustc's async drop test at
// `tests/mir-opt/coroutine/async_drop_shim_usage_with_feature.rs` and
// `tests/mir-opt/coroutine/async_drop_shim_usage_without_feature.rs`.

#![cfg_attr(with_feature, feature(async_drop))]

#![allow(incomplete_features)]

extern crate has_drop_inside_dep;

use has_drop_inside_dep::{with_has_drop, with_has_has_drop};

fn main() {
    println!("size of with_has_drop() = {}", std::mem::size_of_val(&with_has_drop()));
    println!("size of with_has_has_drop() = {}", std::mem::size_of_val(&with_has_has_drop()));
}
