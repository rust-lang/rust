// This test verifies that the MIR we output using the `matches!()` macro is close
// to the MIR for an `if let` branch.

pub enum Foo {
    A,
    B,
    C,
    D,
    E,
    F,
}

// EMIT_MIR matches_macro.issue_77355_opt.PreCodegen.after.mir
fn issue_77355_opt(num: Foo) -> u64 {
    // CHECK-LABEL: fn issue_77355_opt(
    // CHECK: switchInt({{.*}}) -> [1: bb1, 2: bb1, otherwise: bb2];
    // CHECK: bb1: {
    // CHECK-NEXT: _0 = const 23_u64;
    // CHECK-NEXT: return;
    // CHECK: bb2: {
    // CHECK-NEXT: _0 = const 42_u64;
    // CHECK-NEXT: return;
    if matches!(num, Foo::B | Foo::C) { 23 } else { 42 }
}
fn main() {
    issue_77355_opt(Foo::A);
}
