// compile-flags: -O

// EMIT_MIR rustc.main.ConstProp.diff
fn main() {
    let x = (if let Some(true) = Some(true) { 42 } else { 10 }) + 0;
}
