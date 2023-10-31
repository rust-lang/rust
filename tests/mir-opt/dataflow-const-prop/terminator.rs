// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// unit-test: DataflowConstProp

fn foo(n: i32) {}

// EMIT_MIR terminator.main.DataflowConstProp.diff
fn main() {
    let a = 1;
    // Checks that we propagate into terminators.
    foo(a + 1);
}
