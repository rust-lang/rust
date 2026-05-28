// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: Inline
//@ compile-flags: --crate-type=lib

// EMIT_MIR inline_box_fn.call.Inline.diff
fn call(x: Box<dyn Fn(i32)>) {
    // CHECK-LABEL: fn call(
    // CHECK: (inlined <Box<dyn Fn(i32)> as Fn<(i32,)>>::call)
    x(1);
}
