//@ compile-flags: -O -Zmir-opt-level=2
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

// Ensure result.unwrap() gets inlined, but does not generate a landing pad to drop the error.

#![crate_type = "lib"]

// EMIT_MIR result_unwrap.result_unwrap.runtime-optimized.after.mir
pub fn result_unwrap(result: Result<i32, String>) -> i32 {
    // CHECK-LABEL: fn result_unwrap(_1: Result<i32, String>) -> i32
    // CHECK: (inlined #[track_caller] Result::<i32, String>::unwrap)
    // CHECK: = result::unwrap_failed::<String>(
    // CHECK-NOT: (cleanup):
    result.unwrap()
}
