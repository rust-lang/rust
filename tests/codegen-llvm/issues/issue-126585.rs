//@ compile-flags: -Copt-level=s
//@ only-x86_64

// Test for #126585.
// Ensure that this IR doesn't have extra undef phi input, which also guarantees that this asm
// doesn't have subsequent labels and unnecessary `jmp` instructions.

#![crate_type = "lib"]

#[no_mangle]
fn checked_div_round(a: u64, b: u64) -> Option<u64> {
    // CHECK-LABEL: @checked_div_round
    // CHECK: phi
    // CHECK-NOT: undef
    // CHECK: phi
    // CHECK-NOT: undef
    match b {
        0 => None,
        1 => Some(a),
        // `a / b` is computable and `(a % b) * 2` can not overflow since `b >= 2`.
        b => Some(a / b + if (a % b) * 2 >= b { 1 } else { 0 }),
    }
}
