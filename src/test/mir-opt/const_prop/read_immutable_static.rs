// compile-flags: -O

static FOO: u8 = 2;

// EMIT_MIR rustc.main.ConstProp.diff
fn main() {
    let x = FOO + FOO;
}
