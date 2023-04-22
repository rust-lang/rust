// skip-filecheck
// unit-test: ConstProp
// EMIT_MIR cast.main.ConstProp.diff

fn main() {
    let x = 42u8 as u32;
    let y = 42u32 as u8;

    let a = 257;
    let b = a as u8 + 1;
}
