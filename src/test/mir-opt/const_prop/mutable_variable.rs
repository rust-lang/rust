// unit-test
// compile-flags: -O

// EMIT_MIR mutable_variable.main.ConstProp.diff
fn main() {
    let mut x = 42;
    x = 99;
    let y = x;
}
