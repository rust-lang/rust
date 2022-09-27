// unit-test: DataflowConstProp
// compile-flags: -C overflow-checks=on

// EMIT_MIR checked_add.main.DataflowConstProp.diff
fn main() {
    let x: u32 = 1 + 1;
}
