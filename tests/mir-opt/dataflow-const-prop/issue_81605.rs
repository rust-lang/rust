// unit-test: DataflowConstProp

// EMIT_MIR issue_81605.f.DataflowConstProp.diff

// CHECK-LABEL: fn f
fn f() -> usize {
    // CHECK: switchInt(const true) -> [0: {{bb[0-9]+}}, otherwise: {{bb[0-9]+}}];
    1 + if true { 1 } else { 2 }
    // CHECK: _0 = const 2_usize;
}

fn main() {
    f();
}
