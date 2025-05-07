// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: GVN

#[inline(never)]
fn read(_: usize) {}

// EMIT_MIR pointer_expose_provenance.main.GVN.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: [[ptr:_.*]] = const main::FOO;
    // CHECK: [[ref:_.*]] = &raw const (*[[ptr]]);
    // CHECK: [[x:_.*]] = move [[ref]] as usize (PointerExposeProvenance);
    // CHECK: = read(copy [[x]])
    const FOO: &i32 = &1;
    let x = FOO as *const i32 as usize;
    read(x);
}
