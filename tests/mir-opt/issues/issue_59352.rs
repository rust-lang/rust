// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// This test is a mirror of codegen/issue-59352.rs.
// The LLVM inliner doesn't inline `char::method::is_digit()` and so it doesn't recognize this case
// as effectively `if x.is_some() { x.unwrap() } else { 0 }`.
//
// Currently, the MIR optimizer isn't capable of removing the unreachable panic in this test case.
// Once the optimizer can do that, this test case will need to be updated and codegen/issue-59352.rs
// removed.

// EMIT_MIR issue_59352.num_to_digit.PreCodegen.after.mir
// compile-flags: -Z mir-opt-level=3 -Z span_free_formats

pub fn num_to_digit(num: char) -> u32 {
    // CHECK-NOT: panic
    if num.is_digit(8) { num.to_digit(8).unwrap() } else { 0 }
}

pub fn main() {
    num_to_digit('2');
}
