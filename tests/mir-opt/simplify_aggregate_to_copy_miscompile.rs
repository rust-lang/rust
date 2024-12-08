//! The `simplify_aggregate_to_copy` mir-opt introduced in
//! <https://github.com/rust-lang/rust/pull/128299> caused a miscompile because the initial
//! implementation
//!
//! > introduce[d] new dereferences without checking for aliasing
//!
//! This test demonstrates the behavior, and should be adjusted or removed when fixing and relanding
//! the mir-opt.
#![crate_type = "lib"]
// skip-filecheck
//@ compile-flags: -O -Zunsound-mir-opts
//@ test-mir-pass: GVN
#![allow(internal_features)]
#![feature(rustc_attrs, core_intrinsics)]

// EMIT_MIR simplify_aggregate_to_copy_miscompile.foo.GVN.diff
#[no_mangle]
fn foo(v: &mut Option<i32>) -> Option<i32> {
    if let &Some(col) = get(&v) {
        *v = None;
        return Some(col);
    } else {
        unsafe { std::intrinsics::unreachable() }
    }
}

#[no_mangle]
#[inline(never)]
#[rustc_nounwind]
fn get(v: &Option<i32>) -> &Option<i32> {
    v
}
