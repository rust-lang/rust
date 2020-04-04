// compile-flags: -O

// EMIT_MIR rustc.main.ConstProp.diff
fn main() {
    let x: u32 = [42; 8][2] + 0;
}
