// compile-flags: -Cdebug-assertions=false

// EMIT_MIR simplify_pow_of_two_no_overflow_checks.slow_2_u.SimplifyPowOfTwo.diff
fn slow_2_u(a: u32) -> u32 {
    2u32.pow(a)
}

// EMIT_MIR simplify_pow_of_two_no_overflow_checks.slow_2_i.SimplifyPowOfTwo.diff
fn slow_2_i(a: u32) -> i32 {
    2i32.pow(a)
}

// EMIT_MIR simplify_pow_of_two_no_overflow_checks.slow_4_u.SimplifyPowOfTwo.diff
fn slow_4_u(a: u32) -> u32 {
    4u32.pow(a)
}

// EMIT_MIR simplify_pow_of_two_no_overflow_checks.slow_4_i.SimplifyPowOfTwo.diff
fn slow_4_i(a: u32) -> i32 {
    4i32.pow(a)
}

// EMIT_MIR simplify_pow_of_two_no_overflow_checks.slow_256_u.SimplifyPowOfTwo.diff
fn slow_256_u(a: u32) -> u32 {
    256u32.pow(a)
}

// EMIT_MIR simplify_pow_of_two_no_overflow_checks.slow_256_i.SimplifyPowOfTwo.diff
fn slow_256_i(a: u32) -> i32 {
    256i32.pow(a)
}

fn main() {
    slow_2_u(0);
    slow_2_i(0);
    slow_2_u(1);
    slow_2_i(1);
    slow_2_u(2);
    slow_2_i(2);
    slow_4_u(4);
    slow_4_i(4);
    slow_4_u(15);
    slow_4_i(15);
    slow_4_u(16);
    slow_4_i(16);
    slow_4_u(17);
    slow_4_i(17);
    slow_256_u(2);
    slow_256_i(2);
}
