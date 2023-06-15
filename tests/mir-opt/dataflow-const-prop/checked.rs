// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// unit-test: DataflowConstProp
// compile-flags: -Coverflow-checks=on

// EMIT_MIR checked.main.DataflowConstProp.diff
#[allow(arithmetic_overflow)]
fn main() {
    let a = 1;
    let b = 2;
    let c = a + b;

    let d = i32::MAX;
    let e = d + 1;
}
