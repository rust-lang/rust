// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: DataflowConstProp

#[inline(never)]
fn escape<T>(x: &T) {}

#[inline(never)]
fn some_function() {}

// EMIT_MIR ref_without_sb.main.DataflowConstProp.diff
// CHECK-LABEL: fn main(
fn main() {
    // CHECK: debug a => [[a:_.*]];
    // CHECK: debug b => [[b:_.*]];

    let mut a = 0;

    // CHECK: {{_.*}} = escape::<i32>(move {{_.*}}) ->  {{.*}}
    escape(&a);
    a = 1;

    // CHECK: {{_.*}} = some_function() ->  {{.*}}
    some_function();
    // This should currently not be propagated.

    // CHECK-NOT: [[b]] = const
    // CHECK: [[b]] = copy [[a]];
    // CHECK-NOT: [[b]] = const
    let b = a;
}
