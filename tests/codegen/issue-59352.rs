// This test is a mirror of mir-opt/issues/issue-59352.rs. The LLVM inliner doesn't inline
// `char::method::is_digit()` and `char::method::to_digit()`, probably because of their size.
//
// Currently, the MIR optimizer isn't capable of removing the unreachable panic in this test case.
// Once the optimizer can do that, mir-opt/issues/issue-59352.rs will need to be updated and this
// test case should be removed as it will become redundant.

// mir-opt-level=3 enables inlining and enables LLVM to optimize away the unreachable panic call.
// compile-flags: -O -Z mir-opt-level=3

#![crate_type = "rlib"]

// CHECK-LABEL: @num_to_digit
#[no_mangle]
pub fn num_to_digit(num: char) -> u32 {
    // CHECK-NOT: panic
    if num.is_digit(8) { num.to_digit(8).unwrap() } else { 0 }
}
