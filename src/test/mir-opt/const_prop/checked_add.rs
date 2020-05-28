// compile-flags: -C overflow-checks=on

// EMIT_MIR rustc.main.ConstProp.diff
fn main() {
    let x: u32 = 1 + 1;
}
