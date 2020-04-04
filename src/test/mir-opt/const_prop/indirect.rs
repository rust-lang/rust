// compile-flags: -C overflow-checks=on

// EMIT_MIR rustc.main.ConstProp.diff
fn main() {
    let x = (2u32 as u8) + 1;
}
