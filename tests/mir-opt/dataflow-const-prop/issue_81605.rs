//@ test-mir-pass: DataflowConstProp

// EMIT_MIR issue_81605.f.DataflowConstProp.diff

// Plese find the original issue [here](https://github.com/rust-lang/rust/issues/81605).
// This test program comes directly from the issue. Prior to this issue,
// the compiler cannot simplify the return value of `f` into 2. This was
// solved by adding a new MIR constant propagation based on dataflow
// analysis in [#101168](https://github.com/rust-lang/rust/pull/101168).

// CHECK-LABEL: fn f(
fn f() -> usize {
    // CHECK: switchInt(const true) -> [0: {{bb.*}}, otherwise: {{bb.*}}];
    1 + if true { 1 } else { 2 }
    // CHECK: _0 = const 2_usize;
}

fn main() {
    f();
}
