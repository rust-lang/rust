// Checks that inlining threshold can be controlled with
// inline-mir-threshold and inline-hint-threshold options.
//
// compile-flags: -Zinline-mir-threshold=90
// compile-flags: -Zinline-mir-hint-threshold=50

// EMIT_MIR inline_options.main.Inline.after.mir
fn main() {
    not_inlined();
    inlined::<u32>();
}

// Cost is approximately 3 * 25 + 5 = 80.
#[inline]
pub fn not_inlined() { g(); g(); g(); }
pub fn inlined<T>() { g(); g(); g(); }

#[inline(never)]
fn g() {}
