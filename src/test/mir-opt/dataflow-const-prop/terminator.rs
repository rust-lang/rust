// unit-test: DataflowConstProp
// compile-flags: -Zunsound-mir-opts

fn foo(n: i32) {}

// EMIT_MIR terminator.main.DataflowConstProp.diff
fn main() {
    let a = 1;
    foo(a + 1);
}
