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
//
// The lifetime `'a` is redundant (and doesn't print when we print out the type).
// However, we need it so that rustfmt doesn't rip out the `unsafe<>` part for now.
fn propagate() -> unsafe<'a> i32 {
    unsafe {
        let x = 1;
        let binder: unsafe<'a> i32 = wrap_binder!(x);
        binder
    }
}
