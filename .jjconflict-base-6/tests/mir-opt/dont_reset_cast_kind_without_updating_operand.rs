//@ test-mir-pass: GVN
//@ compile-flags: -Zinline-mir --crate-type lib
// EMIT_MIR_FOR_EACH_BIT_WIDTH
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// EMIT_MIR dont_reset_cast_kind_without_updating_operand.test.GVN.diff

fn test() {
    // CHECK-LABEL: fn test(
    // CHECK: debug slf => [[SLF:_.*]];
    // CHECK: debug _x => [[X:_.*]];
    // CHECK: [[X]] = copy [[SLF]] as *mut () (PtrToPtr);
    let vp_ctx: &Box<()> = &Box::new(());
    let slf: *const () = &raw const **vp_ctx;
    let bytes = std::ptr::slice_from_raw_parts(slf, 1);
    let _x = foo(bytes);
}

fn foo(bytes: *const [()]) -> *mut () {
    bytes as *mut ()
}
