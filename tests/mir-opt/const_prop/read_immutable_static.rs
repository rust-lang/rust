// unit-test
// compile-flags: -O

static FOO: u8 = 2;

// EMIT_MIR read_immutable_static.main.ConstProp.diff
fn main() {
    let x = FOO + FOO;
}
