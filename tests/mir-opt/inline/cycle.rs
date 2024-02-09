// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// compile-flags: -Zinline-mir-hint-threshold=1000

// EMIT_MIR cycle.f.Inline.diff
#[inline(always)]
fn f(g: impl Fn()) {
    // CHECK-LABEL: fn f(
    // CHECK-NOT: inlined
    g();
}

// EMIT_MIR cycle.g.Inline.diff
#[inline(always)]
fn g() {
    // CHECK-LABEL: fn g(
    // CHECK-NOT: (inlined f::<fn() {main}>)
    f(main);
}

// EMIT_MIR cycle.main.Inline.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK-NOT: inlined
    // CHECK: (inlined f::<fn() {g}>)
    // CHECK-NOT: inlined
    f(g);
}
