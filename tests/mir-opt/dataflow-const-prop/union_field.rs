// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// Writing to one union field must also invalidate what is known about the other fields,
// since they share the same storage.

//@ test-mir-pass: DataflowConstProp

// EMIT_MIR union_field.main.DataflowConstProp.diff

union U {
    a: u8,
    b: u8,
}

// CHECK-LABEL: fn main(
fn main() {
    // CHECK: debug a => [[a:_.*]];

    let mut u = U { a: 0 };
    u.a = 1;
    u.b = 5;

    // CHECK: [[a]] = copy ({{_.*}}.0: u8);
    let a = unsafe { u.a }; // should not be propagated
}
