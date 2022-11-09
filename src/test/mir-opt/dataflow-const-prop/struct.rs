// unit-test: DataflowConstProp

struct S(i32);

// EMIT_MIR struct.main.DataflowConstProp.diff
fn main() {
    let mut s = S(1);
    let a = s.0 + 2;
    s.0 = 3;
    let b = a + s.0;
}
