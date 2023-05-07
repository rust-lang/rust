// unit-test: ConstProp
// ignore-wasm32 compiled with panic=abort by default
// compile-flags: -Zmir-enable-passes=+NormalizeArrayLen

// EMIT_MIR_FOR_EACH_BIT_WIDTH
// EMIT_MIR bad_op_unsafe_oob_for_slices.main.ConstProp.diff
#[allow(unconditional_panic)]
fn main() {
    let a: *const [_] = &[1, 2, 3];
    unsafe {
        let _b = (*a)[3];
    }
}
