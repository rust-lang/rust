// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ compile-flags: -Zmir-opt-level=1 -Zmir-enable-passes=+MatchBranchSimplification

// EMIT_MIR simplify_locals_fixedpoint.foo.SimplifyLocals-final.diff
fn foo<T>() {
    // CHECK-LABEL: fn foo(
    // CHECK-NOT: let mut {{.*}}: bool;
    // CHECK-NOT: let mut {{.*}}: u8;
    // CHECK-NOT: let mut {{.*}}: bool;
    if let (Some(a), None) = (Option::<u8>::None, Option::<T>::None) {
        if a > 42u8 {}
    }
}

fn main() {
    foo::<()>();
}
