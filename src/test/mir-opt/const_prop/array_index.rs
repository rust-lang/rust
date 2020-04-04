// EMIT_MIR rustc.main.ConstProp.diff

fn main() {
    let x: u32 = [0, 1, 2, 3][2];
}
