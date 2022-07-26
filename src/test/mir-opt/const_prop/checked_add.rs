// unit-test: ConstProp
// compile-flags: -C overflow-checks=on

// EMIT_MIR checked_add.main.ConstProp.diff
fn main() {
    let x: u32 = 1 + 1;
}
