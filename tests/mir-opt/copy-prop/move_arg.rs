// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// Test that we do not move multiple times from the same local.
// unit-test: CopyProp

// EMIT_MIR move_arg.f.CopyProp.diff
pub fn f<T: Copy>(a: T) {
    let b = a;
    g(a, b);
}

#[inline(never)]
pub fn g<T: Copy>(_: T, _: T) {}

fn main() {
    f(5)
}
