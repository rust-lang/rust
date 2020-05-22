// EMIT_MIR rustc.main.ConstProp.diff

fn main() {
    let x = 42u8 as u32;

    let y = 42u32 as u8;
}
