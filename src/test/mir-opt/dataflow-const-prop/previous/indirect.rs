// unit-test: DataflowConstProp
// compile-flags: -C overflow-checks=on

// EMIT_MIR indirect.main.DataflowConstProp.diff
fn main() {
    let x = (2u32 as u8) + 1;
}
