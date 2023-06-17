// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// compile-flags: -Zinline-mir-hint-threshold=1000

// EMIT_MIR cycle.f.Inline.diff
#[inline(always)]
fn f(g: impl Fn()) {
    g();
}

// EMIT_MIR cycle.g.Inline.diff
#[inline(always)]
fn g() {
    f(main);
}

// EMIT_MIR cycle.main.Inline.diff
fn main() {
    f(g);
}
