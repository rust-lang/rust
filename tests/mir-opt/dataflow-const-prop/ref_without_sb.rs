// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// unit-test: DataflowConstProp

#[inline(never)]
fn escape<T>(x: &T) {}

#[inline(never)]
fn some_function() {}

// EMIT_MIR ref_without_sb.main.DataflowConstProp.diff
// CHECK-LABEL: fn main
fn main() {
    // CHECK: debug a => [[a:_.*]];
    // CHECK: debug b => [[b:_.*]];

    let mut a = 0;

    // CHECK: {{_[0-9]+}} = escape::<i32>(move {{_[0-9]+}}) -> [return: {{bb[0-9]+}}, unwind continue];
    escape(&a);
    a = 1;

    // CHECK: {{_[0-9]+}} = some_function() -> [return: {{bb[0-9]+}}, unwind continue];
    some_function();
    // This should currently not be propagated.

    // CHECK: [[b]] = [[a]];
    let b = a;
}
