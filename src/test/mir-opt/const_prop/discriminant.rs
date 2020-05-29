// compile-flags: -O

// EMIT_MIR_FOR_EACH_BIT_WIDTH
// EMIT_MIR rustc.main.ConstProp.diff
fn main() {
    let x = (if let Some(true) = Some(true) { 42 } else { 10 }) + 0;
}
