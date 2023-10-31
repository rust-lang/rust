// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// Check that unreachable code is removed after the destination propagation.
// Regression test for issue #105428.
//
// compile-flags: --crate-type=lib -Zmir-opt-level=0
// compile-flags: -Zmir-enable-passes=+ConstProp,+SimplifyConstCondition-after-const-prop,+DestinationPropagation

// EMIT_MIR unreachable.f.DestinationPropagation.diff
pub fn f<T: Copy>(a: T) {
    let b = a;
    if false {
        g(a, b);
    } else {
        g(b, b);
    }
}

#[inline(never)]
pub fn g<T: Copy>(_: T, _: T) {}
