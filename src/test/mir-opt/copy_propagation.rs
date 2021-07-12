// compile-flags: --crate-type=lib

// EMIT_MIR copy_propagation.write.CopyPropagation.diff
pub fn write<T: Copy>(dst: &mut T, value: T) {
    *dst = value;
}

// EMIT_MIR copy_propagation.id.CopyPropagation.diff
pub fn id<T: Copy>(mut a: T) -> T {
    // Not optimized.
    a = a;
    a
}

// EMIT_MIR copy_propagation.chain.CopyPropagation.diff
pub fn chain<T: Copy>(mut a: T) -> T {
    let mut b;
    let mut c;
    b = a;
    c = b;
    a = c;
    b = a;
    c = b;

    let d = c;
    d
}
