// compile-flags: -O

// EMIT_MIR_FOR_EACH_BIT_WIDTH
// EMIT_MIR rustc.main.ConstProp.diff
fn main() {
    let x: u32 = [42; 8][2] + 0;
}
