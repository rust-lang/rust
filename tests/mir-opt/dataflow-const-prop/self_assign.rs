//@ test-mir-pass: DataflowConstProp

// EMIT_MIR self_assign.main.DataflowConstProp.diff

// CHECK-LABEL: fn main(
fn main() {
    // CHECK: debug a => [[a:_.*]];
    // CHECK: debug b => [[b:_.*]];

    let mut a = 0;

    // CHECK: [[a]] = Add(move {{_.*}}, const 1_i32);
    a = a + 1;

    // CHECK: [[a]] = move {{_.*}};
    a = a;

    // CHECK: [[b]] = &[[a]];
    let mut b = &a;

    // CHECK: [[b]] = move {{_.*}};
    b = b;

    // CHECK: [[a]] = move {{_.*}};
    a = *b;
}
