//@ test-mir-pass: GVN
//@ compile-flags: --crate-type lib

//! Regressions test for a mis-optimization where some functions
//! (`type_id` / `type_name` / `needs_drop`) could be evaluated in
//! a generic context, even though their value depends on some type
//! parameter `T`.
//!
//! In particular, `type_name_of_val(&generic::<T>)` was incorrectly
//! evaluated to the string "crate_name::generic::<T>", and
//! `no_optimize` was incorrectly optimized to `false`.

#![feature(const_type_name)]

fn generic<T>() {}

const fn type_name_contains_i32<T>(_: &T) -> bool {
    let pattern = b"i32";
    let name = std::any::type_name::<T>().as_bytes();
    let mut i = 0;
    'outer: while i < name.len() - pattern.len() + 1 {
        let mut j = 0;
        while j < pattern.len() {
            if name[i + j] != pattern[j] {
                i += 1;
                continue 'outer;
            }
            j += 1;
        }
        return true;
    }
    false
}

// EMIT_MIR gvn_const_eval_polymorphic.optimize_true.GVN.diff
fn optimize_true<T>() -> bool {
    // CHECK-LABEL: fn optimize_true(
    // CHECK: _0 = const true;
    // CHECK-NEXT: return;
    (const { type_name_contains_i32(&generic::<i32>) }) == const { true }
}

// EMIT_MIR gvn_const_eval_polymorphic.optimize_false.GVN.diff
fn optimize_false<T>() -> bool {
    // CHECK-LABEL: fn optimize_false(
    // CHECK: _0 = const false;
    // CHECK-NEXT: return;
    (const { type_name_contains_i32(&generic::<i64>) }) == const { true }
}

// EMIT_MIR gvn_const_eval_polymorphic.no_optimize.GVN.diff
fn no_optimize<T>() -> bool {
    // CHECK-LABEL: fn no_optimize(
    // CHECK: _0 = Eq(const no_optimize::<T>::{constant#0}, const no_optimize::<T>::{constant#1});
    // CHECK-NEXT: return;
    (const { type_name_contains_i32(&generic::<T>) }) == const { true }
}
