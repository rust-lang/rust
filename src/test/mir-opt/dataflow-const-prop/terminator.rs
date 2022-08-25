// unit-test: DataflowConstProp

fn foo(n: i32) {}

// EMIT_MIR terminator.main.DataflowConstProp.diff
fn main() {
    let a = 1;
    foo(a + 1);
}
