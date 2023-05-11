// ignore-wasm32 compiled with panic=abort by default
// unit-test: DataflowConstProp

fn foo(n: i32) {}

// EMIT_MIR terminator.main.DataflowConstProp.diff
fn main() {
    let a = 1;
    // Checks that we propagate into terminators.
    foo(a + 1);
}
