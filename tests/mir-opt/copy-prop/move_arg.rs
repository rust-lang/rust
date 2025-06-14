// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// Test that we do not move multiple times from the same local.
//@ test-mir-pass: CopyProp

// EMIT_MIR move_arg.f.CopyProp.diff
pub fn f<T: Copy>(a: T) {
    // CHECK-LABEL: fn f(
    // CHECK: debug a => [[a:_.*]];
    // CHECK: debug b => [[a]];
    // CHECK: g::<T>(copy [[a]], copy [[a]])
    let b = a;
    g(a, b);
}

#[inline(never)]
pub fn g<T: Copy>(_: T, _: T) {}

fn main() {
    f(5)
}
