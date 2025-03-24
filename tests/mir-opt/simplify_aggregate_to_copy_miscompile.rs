//! The `simplify_aggregate_to_copy` mir-opt introduced in
//! <https://github.com/rust-lang/rust/pull/128299> caused a miscompile because the initial
//! implementation
//!
//! > introduce[d] new dereferences without checking for aliasing
//!
//! This test demonstrates the behavior, and should be adjusted or removed when fixing and relanding
//! the mir-opt.
#![crate_type = "lib"]
//@ test-mir-pass: GVN
#![allow(internal_features)]
#![feature(rustc_attrs, core_intrinsics)]

// EMIT_MIR simplify_aggregate_to_copy_miscompile.foo.GVN.diff
#[no_mangle]
fn foo(v: &mut Option<i32>) -> Option<i32> {
    // CHECK-LABEL: fn foo(
    // CHECK-SAME: [[v:_.*]]: &mut Option<i32>
    // CHECK: [[v_alias_1:_.*]] = &(*_1)
    // CHECK-NEXT: [[v_alias_2:_.*]] = get(move [[v_alias_1]])
    // CHECK: (*[[v]]) = const Option::<i32>::None;
    // CHECK-NOT: _0 = copy (*[[v_alias_2]])
    // CHECK: _0 = Option::<i32>::Some
    // CHECK-NOT: _0 = copy (*[[v_alias_2]])
    if let &Some(col) = get(v) {
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
