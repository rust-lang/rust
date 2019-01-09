#![feature(const_int_wrapping)]

const ADD_A: u32 = 200u32.wrapping_add(55);
const ADD_B: u32 = 200u32.wrapping_add(u32::max_value());

const SUB_A: u32 = 100u32.wrapping_sub(100);
const SUB_B: u32 = 100u32.wrapping_sub(u32::max_value());

const MUL_A: u8 = 10u8.wrapping_mul(12);
const MUL_B: u8 = 25u8.wrapping_mul(12);

const SHL_A: u32 = 1u32.wrapping_shl(7);
const SHL_B: u32 = 1u32.wrapping_shl(128);

const SHR_A: u32 = 128u32.wrapping_shr(7);
const SHR_B: u32 = 128u32.wrapping_shr(128);

fn ident<T>(ident: T) -> T {
    ident
}

fn main() {
    assert_eq!(ADD_A, ident(255));
    assert_eq!(ADD_B, ident(199));

    assert_eq!(SUB_A, ident(0));
    assert_eq!(SUB_B, ident(101));

    assert_eq!(MUL_A, ident(120));
    assert_eq!(MUL_B, ident(44));

    assert_eq!(SHL_A, ident(128));
    assert_eq!(SHL_B, ident(1));

    assert_eq!(SHR_A, ident(1));
    assert_eq!(SHR_B, ident(128));
}
