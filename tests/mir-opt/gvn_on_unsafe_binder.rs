// skip-filecheck
//@ test-mir-pass: GVN

// EMIT_MIR gvn_on_unsafe_binder.test.GVN.diff
// EMIT_MIR gvn_on_unsafe_binder.propagate.GVN.diff

#![feature(unsafe_binders)]

use std::unsafe_binder::wrap_binder;

// Test for ICE <https://github.com/rust-lang/rust/issues/137846>.
fn test() {
    unsafe {
        let x = 1;
        let binder: unsafe<'a> &'a i32 = wrap_binder!(&x);
    }
}

// Test that GVN propagates const values through unsafe binders.
fn propagate() -> unsafe<> i32 {
    unsafe {
        let x = 1;
        let binder: unsafe<> i32 = wrap_binder!(x);
        binder
    }
}
