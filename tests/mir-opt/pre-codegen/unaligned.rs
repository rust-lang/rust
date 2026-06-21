//@ compile-flags: -O -Zmir-opt-level=2

#![crate_type = "lib"]

// EMIT_MIR unaligned.unaligned_typed_copy.runtime-optimized.after.mir
pub unsafe fn unaligned_typed_copy(src: *const String, dst: *mut String) {
    // CHECK-LABEL: fn unaligned_typed_copy(
    // CHECK: debug src => _1;
    // CHECK: debug dst => _2;
    // CHECK: debug val => [[VAL:_.+]];
    // CHECK: [[SRCU:_.+]] = copy _1 as *const std::ptr::Packed<std::string::String> (PtrToPtr);
    // CHECK: [[VAL]] = copy ((*[[SRCU]]).0: std::string::String);
    // CHECK: [[DSTU:_.+]] = copy _2 as *mut std::ptr::Packed<std::string::String> (PtrToPtr);
    // CHECK: ((*[[DSTU]]).0: std::string::String) = copy [[VAL]];
    // CHECK-NOT: drop
    unsafe {
        let val = std::ptr::read_unaligned(src);
        std::ptr::write_unaligned(dst, val);
    }
}
