// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// Writing to one union field must also invalidate what is known about the other fields,
// since they share the same storage.

//@ test-mir-pass: DataflowConstProp
//@ compile-flags: --crate-type=lib

union U {
    a: u8,
    b: u8,
}

// EMIT_MIR union_field.read_other_field.DataflowConstProp.diff
pub fn read_other_field() -> u8 {
    // CHECK-LABEL: fn read_other_field(
    // CHECK: _0 = copy ({{_.*}}.0: u8);
    let mut u = U { a: 0 };
    u.a = 1;
    u.b = 5;
    unsafe { u.a }
}
