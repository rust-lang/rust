//@ compile-flags: -Copt-level=3
//@ only-x86_64
#![crate_type = "lib"]

// Ensure that the optimized variant of the function gets auto-vectorized and
// that the inner helper function is inlined.
// CHECK-LABEL: @eq_ignore_ascii_case_autovectorized
#[no_mangle]
pub fn eq_ignore_ascii_case_autovectorized(s: &str, other: &str) -> bool {
    // CHECK: load <16 x i8>
    // CHECK: load <16 x i8>
    // CHECK: bitcast <16 x i1>
    // CHECK-NOT: call {{.*}}eq_ignore_ascii_inner
    // CHECK-NOT: panic
    s.eq_ignore_ascii_case(other)
}
