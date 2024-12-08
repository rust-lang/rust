//@ test-mir-pass: GVN
// EMIT_MIR reify_fn_ptr.main.GVN.diff

fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: [[ptr:_.*]] = main as fn() (PointerCoercion(ReifyFnPointer, AsCast));
    // CHECK: [[addr:_.*]] = move [[ptr]] as usize (PointerExposeProvenance);
    // CHECK: [[back:_.*]] = move [[addr]] as *const fn() (PointerWithExposedProvenance);
    let _ = main as usize as *const fn();
}
