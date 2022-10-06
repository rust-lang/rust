// unit-test: DataflowConstProp
// compile-flags: -Zunsound-mir-opts

// EMIT_MIR cast.main.DataflowConstProp.diff
fn main() {
    let a = 257;
    let b = *&a as u8 + 1;
}
