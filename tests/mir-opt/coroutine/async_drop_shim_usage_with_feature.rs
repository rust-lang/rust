//@ edition: 2024
//@ skip-filecheck
//@ aux-build: has_drop_inside_dep.rs

// WARNING: If you would ever want to modify this test,
// please consider modifying rustc's async drop test at
// `tests/ui/async-await/async-drop/async-drop-shim-usage.rs`.

// FIXME: This test is a variant of asynd-drop-shim-usage on MIR,
// but compiletest seems it cannot produce EMIR_MIR in revisions.
// This test corresponds to `with_feature` revision,
// and other `without_feature` revision could found at
// `tests/mir-opt/coroutine/async_drop_shim_usage_without_feature.rs`.

#![feature(async_drop)]
#![allow(incomplete_features)]

extern crate has_drop_inside_dep;

use has_drop_inside_dep::{with_has_drop, with_has_has_drop};

// EMIT_MIR core.future-async_drop-async_drop_in_place-{closure#0}.HasDrop.coroutine_before.0.mir
// EMIT_MIR core.future-async_drop-async_drop_in_place-{closure#0}.HasHasDrop.coroutine_before.0.mir
#[allow(unused)]
fn main() {
    println!("size of with_has_drop() = {}", std::mem::size_of_val(&with_has_drop()));
    println!("size of with_has_has_drop() = {}", std::mem::size_of_val(&with_has_has_drop()));
}
