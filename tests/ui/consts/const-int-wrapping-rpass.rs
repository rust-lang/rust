//@ run-pass

const ADD_A: u32 = 200u32.wrapping_add(55);
const ADD_B: u32 = 200u32.wrapping_add(u32::MAX);

const SUB_A: u32 = 100u32.wrapping_sub(100);
const SUB_B: u32 = 100u32.wrapping_sub(u32::MAX);

const MUL_A: u8 = 10u8.wrapping_mul(12);
const MUL_B: u8 = 25u8.wrapping_mul(12);

const SHL_A: u32 = 1u32.wrapping_shl(7);
const SHL_B: u32 = 1u32.wrapping_shl(128);

const SHR_A: u32 = 128u32.wrapping_shr(7);
const SHR_B: u32 = 128u32.wrapping_shr(128);

const NEG_A: u32 = 5u32.wrapping_neg();
const NEG_B: u32 = 1234567890u32.wrapping_neg();

const ABS_POS: i32 = 10i32.wrapping_abs();
const ABS_NEG: i32 = (-10i32).wrapping_abs();
const ABS_MIN: i32 = i32::MIN.wrapping_abs();

fn main() {
    assert_eq!(ADD_A, 255);
    assert_eq!(ADD_B, 199);

    assert_eq!(SUB_A, 0);
    assert_eq!(SUB_B, 101);

    assert_eq!(MUL_A, 120);
    assert_eq!(MUL_B, 44);

    assert_eq!(SHL_A, 128);
    assert_eq!(SHL_B, 1);

    assert_eq!(SHR_A, 1);
    assert_eq!(SHR_B, 128);

    assert_eq!(NEG_A, 4294967291);
    assert_eq!(NEG_B, 3060399406);

    assert_eq!(ABS_POS, 10);
    assert_eq!(ABS_NEG, 10);
    assert_eq!(ABS_MIN, i32::MIN);
}
