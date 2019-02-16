const ADD_A: (u32, bool) = 5u32.overflowing_add(2);
const ADD_B: (u32, bool) = u32::max_value().overflowing_add(1);

const SUB_A: (u32, bool) = 5u32.overflowing_sub(2);
const SUB_B: (u32, bool) = 0u32.overflowing_sub(1);

const MUL_A: (u32, bool) = 5u32.overflowing_mul(2);
const MUL_B: (u32, bool) = 1_000_000_000u32.overflowing_mul(10);

const SHL_A: (u32, bool) = 0x1u32.overflowing_shl(4);
const SHL_B: (u32, bool) = 0x1u32.overflowing_shl(132);

const SHR_A: (u32, bool) = 0x10u32.overflowing_shr(4);
const SHR_B: (u32, bool) = 0x10u32.overflowing_shr(132);

fn ident<T>(ident: T) -> T {
    ident
}

fn main() {
    assert_eq!(ADD_A, ident((7, false)));
    assert_eq!(ADD_B, ident((0, true)));

    assert_eq!(SUB_A, ident((3, false)));
    assert_eq!(SUB_B, ident((u32::max_value(), true)));

    assert_eq!(MUL_A, ident((10, false)));
    assert_eq!(MUL_B, ident((1410065408, true)));

    assert_eq!(SHL_A, ident((0x10, false)));
    assert_eq!(SHL_B, ident((0x10, true)));

    assert_eq!(SHR_A, ident((0x1, false)));
    assert_eq!(SHR_B, ident((0x1, true)));
}
