//@ test-mir-pass: DataflowConstProp
//@ compile-flags: -Zdump-mir-exclude-alloc-bytes
// EMIT_MIR_FOR_EACH_BIT_WIDTH

// EMIT_MIR tuple.main.DataflowConstProp.diff

// CHECK-LABEL: fn main(
fn main() {
    // CHECK: debug a => [[a:_.*]];
    // CHECK: debug b => [[b:_.*]];
    // CHECK: debug c => [[c:_.*]];
    // CHECK: debug d => [[d:_.*]];

    // CHECK: [[a]] = const (1_i32, 2_i32);
    let mut a = (1, 2);

    // CHECK: [[b]] = const 6_i32;
    let b = a.0 + a.1 + 3;

    // CHECK: [[a]] = const (2_i32, 3_i32);
    a = (2, 3);

    // CHECK: [[c]] = const 11_i32;
    let c = a.0 + a.1 + b;

    // CHECK: [[d]] = (const 6_i32, const (2_i32, 3_i32), const 11_i32);
    let d = (b, a, c);
}
