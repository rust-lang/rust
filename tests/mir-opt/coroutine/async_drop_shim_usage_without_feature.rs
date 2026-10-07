//@ edition: 2024
//@ skip-filecheck
//@ aux-build: has_drop_inside_dep.rs

// WARNING: This test is false positive! If this test is passed,
// referred issue is reproducing. Otherwise, referred issue is solved.

// FIXME: This test might be working improperly,
// because compiletest seems it doesn't care about
// where dumped MIR came from, includes aux-build[s].
// This test excepts EMIT_MIR grabs MIR on compilation
// of this file, not auxiliary file.

// NOTE: This file below is hand-written! This may produces incorrect result.
// `tests/mir-opt/coroutine/async_drop_shim_usage_without_feature.*.HasHasDrop.*.mir`

// WARNING: If you would ever want to modify this test,
// please consider modifying rustc's async drop test at
// `tests/ui/async-await/async-drop/async-drop-shim-usage.rs`.

// FIXME: This test is a variant of asynd-drop-shim-usage on MIR,
// but compiletest seems it cannot produce EMIR_MIR in revisions.
// This test corresponds to `without_feature` revision,
// and other `with_feature` revision could found at
// `tests/mir-opt/coroutine/async_drop_shim_usage_with_feature.rs`.

extern crate has_drop_inside_dep;

use has_drop_inside_dep::{with_has_drop, with_has_has_drop};

// EMIT_MIR core.future-async_drop-async_drop_in_place-{closure#0}.HasDrop.coroutine_before.0.mir
// EMIT_MIR core.future-async_drop-async_drop_in_place-{closure#0}.HasHasDrop.coroutine_before.0.mir
#[allow(unused)]
fn main() {
    println!("size of with_has_drop() = {}", std::mem::size_of_val(&with_has_drop()));
    println!("size of with_has_has_drop() = {}", std::mem::size_of_val(&with_has_has_drop()));
}
