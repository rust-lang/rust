use int::LargeInt;
use int::Int;

trait UAdd: LargeInt {
    fn uadd(self, other: Self) -> Self {
        let (low, carry) = self.low().overflowing_add(other.low());
        let high = self.high().wrapping_add(other.high());
        let carry = if carry { Self::HighHalf::ONE } else { Self::HighHalf::ZERO };
        Self::from_parts(low, high.wrapping_add(carry))
    }
}

impl UAdd for u128 {}

trait Add: Int
    where <Self as Int>::UnsignedInt: UAdd
{
    fn add(self, other: Self) -> Self {
        Self::from_unsigned(self.unsigned().uadd(other.unsigned()))
    }
}

impl Add for u128 {}
impl Add for i128 {}

trait Addo: Add
    where <Self as Int>::UnsignedInt: UAdd
{
    fn addo(self, other: Self, overflow: &mut i32) -> Self {
        *overflow = 0;
        let result = Add::add(self, other);
        if other >= Self::ZERO {
            if result < self {
                *overflow = 1;
            }
        } else {
            if result >= self {
                *overflow = 1;
            }
        }
        result
    }
}

impl Addo for i128 {}
impl Addo for u128 {}

#[cfg_attr(not(stage0), lang = "i128_add")]
pub fn rust_i128_add(a: i128, b: i128) -> i128 {
    rust_u128_add(a as _, b as _) as _
}
#[cfg_attr(not(stage0), lang = "i128_addo")]
pub fn rust_i128_addo(a: i128, b: i128) -> (i128, bool) {
    let mut oflow = 0;
    let r = a.addo(b, &mut oflow);
    (r, oflow != 0)
}
#[cfg_attr(not(stage0), lang = "u128_add")]
pub fn rust_u128_add(a: u128, b: u128) -> u128 {
    a.add(b)
}
#[cfg_attr(not(stage0), lang = "u128_addo")]
pub fn rust_u128_addo(a: u128, b: u128) -> (u128, bool) {
    let mut oflow = 0;
    let r = a.addo(b, &mut oflow);
    (r, oflow != 0)
}

#[test]
fn test_add() {
    assert_eq!(rust_u128_add(1, 2), 3);
    assert_eq!(rust_u128_add(!0, 3), 2);
    assert_eq!(rust_u128_add(1 << 63, 1 << 63), 1 << 64);
    assert_eq!(rust_u128_add(
        0x54009B79B43145A0_B781BF1FD491296E_u128,
        0x6019CEECA5354210_839AB51D155FF7F3_u128),
        0xB41A6A66596687B1_3B1C743CE9F12161_u128);
    assert_eq!(rust_u128_add(
        0x3AE89C3AACEE47CD_8721275248B38DDB_u128,
        0xEFDD73C41D344744_B0842900C3352A63_u128),
        0x2AC60FFECA228F12_37A550530BE8B83E_u128);

    assert_eq!(rust_i128_add(1, 2), 3);
    assert_eq!(rust_i128_add(-1, 3), 2);
}

#[test]
fn test_addo() {
    assert_eq!(rust_u128_addo(1, 2), (3, false));
    assert_eq!(rust_u128_addo(!0, 3), (2, true));
    assert_eq!(rust_u128_addo(1 << 63, 1 << 63), (1 << 64, false));
    assert_eq!(rust_u128_addo(
        0x54009B79B43145A0_B781BF1FD491296E_u128,
        0x6019CEECA5354210_839AB51D155FF7F3_u128),
        (0xB41A6A66596687B1_3B1C743CE9F12161_u128, false));
    assert_eq!(rust_u128_addo(
        0x3AE89C3AACEE47CD_8721275248B38DDB_u128,
        0xEFDD73C41D344744_B0842900C3352A63_u128),
        (0x2AC60FFECA228F12_37A550530BE8B83E_u128, true));

    assert_eq!(rust_i128_addo(1, 2), (3, false));
    assert_eq!(rust_i128_addo(-1, 3), (2, false));
    assert_eq!(rust_i128_addo(1 << 63, 1 << 63), (1 << 64, false));
    assert_eq!(rust_i128_addo(
        0x54009B79B43145A0_B781BF1FD491296E_i128,
        0x6019CEECA5354210_839AB51D155FF7F3_i128),
        (-0x4BE59599A699784E_C4E38BC3160EDE9F_i128, true));
    assert_eq!(rust_i128_addo(
        0x3AE89C3AACEE47CD_8721275248B38DDB_i128,
        -0x10228C3BE2CBB8BB_4F7BD6FF3CCAD59D_i128),
        (0x2AC60FFECA228F12_37A550530BE8B83E_i128, false));
    assert_eq!(rust_i128_addo(
        -0x54009B79B43145A0_B781BF1FD491296E_i128,
        -0x6019CEECA5354210_839AB51D155FF7F3_i128),
        (0x4BE59599A699784E_C4E38BC3160EDE9F_i128, true));
    assert_eq!(rust_i128_addo(
        -0x3AE89C3AACEE47CD_8721275248B38DDB_i128,
        0x10228C3BE2CBB8BB_4F7BD6FF3CCAD59D_i128),
        (-0x2AC60FFECA228F12_37A550530BE8B83E_i128, false));
}