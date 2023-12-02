// unit-test: ConstProp
// EMIT_MIR reify_fn_ptr.main.ConstProp.diff

fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: [[ptr:_.*]] = main as fn() (PointerCoercion(ReifyFnPointer));
    // CHECK: [[addr:_.*]] = move [[ptr]] as usize (PointerExposeAddress);
    // CHECK: [[back:_.*]] = move [[addr]] as *const fn() (PointerFromExposedAddress);
    let _ = main as usize as *const fn();
}
