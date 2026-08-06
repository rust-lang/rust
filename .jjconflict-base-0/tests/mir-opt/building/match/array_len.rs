// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ compile-flags: -Zmir-opt-level=0

fn opaque<T>(x: T) {}

// EMIT_MIR array_len.const_array_len.built.after.mir
fn const_array_len<T>(x: [T; 5]) {
    // CHECK-LABEL: fn const_array_len(
    // CHECK-NOT: Len
    // CHECK-NOT: PtrMetadata
    // CHECK: = const 5_usize;
    if let [a, b, rest @ .., e] = x {
        opaque(a);
        opaque(b);
        opaque(rest);
        opaque(e);
    }
}

// EMIT_MIR array_len.slice_len.built.after.mir
fn slice_len<T>(x: &[T]) {
    // CHECK-LABEL: fn slice_len(
    // CHECK-NOT: Len
    // CHECK: = PtrMetadata(copy _1);
    if let [a, b, rest @ .., e] = x {
        opaque(a);
        opaque(b);
        opaque(rest);
        opaque(e);
    }
}
