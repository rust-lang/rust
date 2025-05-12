//@ test-mir-pass: DataflowConstProp

// EMIT_MIR if.main.DataflowConstProp.diff
// CHECK-LABEL: fn main(
fn main() {
    // CHECK: debug b => [[b:_.*]];
    // CHECK: debug c => [[c:_.*]];
    // CHECK: debug d => [[d:_.*]];
    // CHECK: debug e => [[e:_.*]];

    let a = 1;

    // CHECK: switchInt(const true) -> [0: {{bb.*}}, otherwise: {{bb.*}}];
    // CHECK: [[b]] = const 2_i32;
    let b = if a == 1 { 2 } else { 3 };

    // CHECK: [[c]] = const 3_i32;
    let c = b + 1;

    // CHECK: switchInt(const true) -> [0: {{bb.*}}, otherwise: {{bb.*}}];
    // CHECK: [[d]] = const 1_i32;
    let d = if a == 1 { a } else { a + 1 };

    // CHECK: [[e]] = const 2_i32;
    let e = d + 1;
}
