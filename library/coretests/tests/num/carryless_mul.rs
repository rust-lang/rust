//! Tests the `Unsigned::{carryless_mul, widening_carryless_mul, carrying_carryless_mul}` methods.

#[test]
fn carryless_mul_u128() {
    assert_eq_const_safe!(u128: <u128>::carryless_mul(0, 0), 0);
    assert_eq_const_safe!(u128: <u128>::carryless_mul(1, 1), 1);

    assert_eq_const_safe!(
        u128: <u128>::carryless_mul(
            0x0123456789ABCDEF_FEDCBA9876543210,
            1u128 << 64,
        ),
        0xFEDCBA9876543210_0000000000000000
    );

    assert_eq_const_safe!(
        u128: <u128>::carryless_mul(
            0x0123456789ABCDEF_FEDCBA9876543210,
            (1u128 << 64) | 1,
        ),
        0xFFFFFFFFFFFFFFFF_FEDCBA9876543210
    );

    assert_eq_const_safe!(
        u128: <u128>::carryless_mul(
            0x0123456789ABCDEF_FEDCBA9876543211,
            1u128 << 127,
        ),
        0x8000000000000000_0000000000000000
    );

    assert_eq_const_safe!(
        u128: <u128>::carryless_mul(
            0xAAAAAAAAAAAAAAAA_AAAAAAAAAAAAAAAA,
            0x5555555555555555_5555555555555555,
        ),
        0x2222222222222222_2222222222222222
    );

    assert_eq_const_safe!(
        u128: <u128>::carryless_mul(
            (1 << 127) | (1 << 64) | 1,
            (1 << 63) | 1
        ),
        (1 << 64) | (1 << 63) | 1
    );

    assert_eq_const_safe!(
        u128: <u128>::carryless_mul(
            0x8000000000000000_0000000000000001,
            0x7FFFFFFFFFFFFFFF_FFFFFFFFFFFFFFFF,
        ),
        0xFFFFFFFFFFFFFFFF_FFFFFFFFFFFFFFFF
    );
}

#[test]
fn carryless_mul_u64() {
    assert_eq_const_safe!(u64: <u64>::carryless_mul(0, 0), 0);
    assert_eq_const_safe!(u64: <u64>::carryless_mul(1, 1), 1);

    assert_eq_const_safe!(
        u64: <u64>::carryless_mul(
            0x0123_4567_89AB_CDEF,
            1u64 << 32,
        ),
        0x89AB_CDEF_0000_0000
    );

    assert_eq_const_safe!(
        u64: <u64>::carryless_mul(
            0x0123_4567_89AB_CDEF,
            (1u64 << 32) | 1,
        ),
        0x8888_8888_89AB_CDEF
    );

    assert_eq_const_safe!(
        u64: <u64>::carryless_mul(
            0x0123_4567_89AB_CDEF,
            1u64 << 63,
        ),
        0x8000_0000_0000_0000
    );

    assert_eq_const_safe!(
        u64: <u64>::carryless_mul(
            0xAAAA_AAAA_AAAA_AAAA,
            0x5555_5555_5555_5555,
        ),
        0x2222_2222_2222_2222
    );

    assert_eq_const_safe!(
        u64: <u64>::carryless_mul(
            (1u64 << 63) | (1u64 << 32) | 1,
            (1u64 << 31) | 1,
        ),
          (1u64 << 32) | (1u64 << 31) | 1
    );

    assert_eq_const_safe!(
        u64: <u64>::carryless_mul(
            0x8000_0000_0000_0001,
            0x7FFF_FFFF_FFFF_FFFF,
        ),
        0xFFFF_FFFF_FFFF_FFFF
    );
}

#[test]
fn carryless_mul_u32() {
    assert_eq_const_safe!(
        u32: <u32>::carryless_mul(0x0123_4567, 1u32 << 16),
        0x4567_0000
    );

    assert_eq_const_safe!(
        u32: <u32>::carryless_mul(0xAAAA_AAAA, 0x5555_5555),
        0x2222_2222
    );
}

#[test]
fn carryless_mul_u16() {
    assert_eq_const_safe!(
        u16: <u16>::carryless_mul(0x0123, 1u16 << 8),
        0x2300
    );

    assert_eq_const_safe!(
        u16: <u16>::carryless_mul(0xAAAA, 0x5555),
        0x2222
    );
}

#[test]
fn carryless_mul_u8() {
    assert_eq_const_safe!(
        u8: <u8>::carryless_mul(0x01, 1u8 << 4),
        0x10
    );

    assert_eq_const_safe!(
        u8: <u8>::carryless_mul(0xAA, 0x55),
        0x22
    );
}

#[test]
fn widening_carryless_mul() {
    assert_eq_const_safe!(
        u16: <u8>::widening_carryless_mul(0xEFu8, 1u8 << 7),
        0x7780u16
    );
    assert_eq_const_safe!(
        u16: <u8>::widening_carryless_mul(0xEFu8, (1u8 << 7) | 1),
        0x776Fu16
    );

    assert_eq_const_safe!(
        u32: <u16>::widening_carryless_mul(0xBEEFu16, 1u16 << 15),
        0x5F77_8000u32
    );
    assert_eq_const_safe!(
        u32: <u16>::widening_carryless_mul(0xBEEFu16, (1u16 << 15) | 1),
        0x5F77_3EEFu32
    );

    assert_eq_const_safe!(
        u64: <u32>::widening_carryless_mul(0xDEAD_BEEFu32, 1u32 << 31),
        0x6F56_DF77_8000_0000u64
    );
    assert_eq_const_safe!(
        u64: <u32>::widening_carryless_mul(0xDEAD_BEEFu32, (1u32 << 31) | 1),
        0x6F56_DF77_5EAD_BEEFu64
    );

    assert_eq_const_safe!(
        u128: <u64>::widening_carryless_mul(0xDEAD_BEEF_FACE_FEEDu64, 1u64 << 63),
        147995377545877439359040026616086396928

    );
    assert_eq_const_safe!(
        u128: <u64>::widening_carryless_mul(0xDEAD_BEEF_FACE_FEEDu64, (1u64 << 63) | 1),
        147995377545877439356638973527682121453
    );
}

#[test]
fn carrying_carryless_mul() {
    assert_eq_const_safe!(
        (u8, u8): <u8>::carrying_carryless_mul(0xEFu8, 1u8 << 7, 0),
        (0x80u8, 0x77u8)
    );
    assert_eq_const_safe!(
        (u8, u8): <u8>::carrying_carryless_mul(0xEFu8, (1u8 << 7) | 1, 0xEF),
        (0x80u8, 0x77u8)
    );

    assert_eq_const_safe!(
        (u16, u16): <u16>::carrying_carryless_mul(0xBEEFu16, 1u16 << 15, 0),
        (0x8000u16, 0x5F77u16)
    );
    assert_eq_const_safe!(
        (u16, u16): <u16>::carrying_carryless_mul(0xBEEFu16, (1u16 << 15) | 1, 0xBEEF),
        (0x8000u16, 0x5F77u16)
    );

    assert_eq_const_safe!(
        (u32, u32): <u32>::carrying_carryless_mul(0xDEAD_BEEFu32, 1u32 << 31, 0),
        (0x8000_0000u32, 0x6F56_DF77u32)
    );
    assert_eq_const_safe!(
        (u32, u32): <u32>::carrying_carryless_mul(0xDEAD_BEEFu32, (1u32 << 31) | 1, 0xDEAD_BEEF),
        (0x8000_0000u32, 0x6F56_DF77u32)
    );

    assert_eq_const_safe!(
        (u64, u64): <u64>::carrying_carryless_mul(0xDEAD_BEEF_FACE_FEEDu64, 1u64 << 63, 0),
        (9223372036854775808, 8022845492652638070)
    );
    assert_eq_const_safe!(
        (u64, u64): <u64>::carrying_carryless_mul(
            0xDEAD_BEEF_FACE_FEEDu64,
            (1u64 << 63) | 1,
            0xDEAD_BEEF_FACE_FEED,
        ),
        (9223372036854775808, 8022845492652638070)
    );

    assert_eq_const_safe!(
        (u128, u128): <u128>::carrying_carryless_mul(
            0xDEAD_BEEF_FACE_FEED_0123_4567_89AB_CDEFu128,
            1u128 << 127,
            0,
        ),
        (
            0x8000_0000_0000_0000_0000_0000_0000_0000u128,
            147995377545877439359081019380694640375,
        )
    );
    assert_eq_const_safe!(
        (u128, u128): <u128>::carrying_carryless_mul(
            0xDEAD_BEEF_FACE_FEED_0123_4567_89AB_CDEFu128,
            (1u128 << 127) | 1,
            0xDEAD_BEEF_FACE_FEED_0123_4567_89AB_CDEF,
        ),
        (
            0x8000_0000_0000_0000_0000_0000_0000_0000u128,
            147995377545877439359081019380694640375,
        )
    );
}
