// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ compile-flags: -O --crate-type lib

// To avoid MIR blow-up, don't inline large callees into simple shim callers,
// but *do* inline other trivial things.

extern "Rust" {
    fn other_thing(x: i32);
}

#[inline]
unsafe fn call_twice(x: i32) {
    unsafe {
        other_thing(x);
        other_thing(x);
    }
}

// EMIT_MIR inline_more_in_non_inline.monomorphic_not_inline.Inline.after.mir
#[no_mangle]
pub unsafe fn monomorphic_not_inline(x: i32) {
    // CHECK-LABEL: monomorphic_not_inline
    // CHECK: other_thing
    // CHECK: other_thing
    unsafe { call_twice(x) };
}

// EMIT_MIR inline_more_in_non_inline.marked_inline_direct.Inline.after.mir
#[inline]
pub unsafe fn marked_inline_direct(x: i32) {
    // CHECK-LABEL: marked_inline_direct
    // CHECK-NOT: other_thing
    // CHECK: call_twice
    // CHECK-NOT: other_thing
    unsafe { call_twice(x) };
}

// EMIT_MIR inline_more_in_non_inline.marked_inline_indirect.Inline.after.mir
#[inline]
pub unsafe fn marked_inline_indirect(x: i32) {
    // CHECK-LABEL: marked_inline_indirect
    // CHECK-NOT: other_thing
    // CHECK: call_twice
    // CHECK-NOT: other_thing
    unsafe { marked_inline_direct(x) };
}
