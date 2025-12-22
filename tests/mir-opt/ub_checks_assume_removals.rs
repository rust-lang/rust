//@ test-mir-pass: SimplifyCfg-final
//@ compile-flags: -Zmir-opt-level=1

#![crate_type = "lib"]
#![feature(core_intrinsics)]

// This test ensures that `assume(RuntimeChecks)`
// is not emitted in MIR after SimplifyCfg.

// CHECK-NOT: assume(RuntimeChecks)
// CHECK-NOT: RuntimeChecks

// EMIT_MIR_FOR_EACH ub_checks_assume_removals.test SimplifyCfg-final
pub unsafe fn test(ptr: *const u8) -> u8 {
    ptr.read()
}

// EMIT_MIR_FOR_EACH ub_checks_assume_removals.with_unchecked SimplifyCfg-final
pub unsafe fn with_unchecked(x: i32, y: i32) -> i32 {
    x.unchecked_add(y)
}
