use int::LargeInt;

trait Sub: LargeInt {
    fn sub(self, other: Self) -> Self {
        let neg_other = (!other).wrapping_add(Self::ONE);
        self.wrapping_add(neg_other)
    }
}

impl Sub for i128 {}
impl Sub for u128 {}

trait Subo: Sub {
    fn subo(self, other: Self, overflow: &mut i32) -> Self {
        *overflow = 0;
        let result = Sub::sub(self, other);
        if other >= Self::ZERO {
            if result > self {
                *overflow = 1;
            }
        } else {
            if result <= self {
                *overflow = 1;
            }
        }
        result
    }
}

impl Subo for i128 {}
impl Subo for u128 {}

#[cfg_attr(not(stage0), lang = "i128_sub")]
#[allow(dead_code)]
fn rust_i128_sub(a: i128, b: i128) -> i128 {
    rust_u128_sub(a as _, b as _) as _
}
#[cfg_attr(not(stage0), lang = "i128_subo")]
#[allow(dead_code)]
fn rust_i128_subo(a: i128, b: i128) -> (i128, bool) {
    let mut oflow = 0;
    let r = a.subo(b, &mut oflow);
    (r, oflow != 0)
}
#[cfg_attr(not(stage0), lang = "u128_sub")]
#[allow(dead_code)]
fn rust_u128_sub(a: u128, b: u128) -> u128 {
    a.sub(b)
}
#[cfg_attr(not(stage0), lang = "u128_subo")]
#[allow(dead_code)]
fn rust_u128_subo(a: u128, b: u128) -> (u128, bool) {
    let mut oflow = 0;
    let r = a.subo(b, &mut oflow);
    (r, oflow != 0)
}

#[test]
fn test_sub() {
    assert_eq!(rust_u128_sub(3, 2), 1);
    assert_eq!(rust_u128_sub(2, 3), !0);
    assert_eq!(rust_u128_sub(1 << 64, 1 << 63), 1 << 63);
    assert_eq!(rust_u128_sub(
        0xB41A6A66596687B1_3B1C743CE9F12161_u128,
        0x6019CEECA5354210_839AB51D155FF7F3_u128),
        0x54009B79B43145A0_B781BF1FD491296E_u128);
    assert_eq!(rust_u128_sub(
        0x2AC60FFECA228F12_37A550530BE8B83E_u128,
        0xEFDD73C41D344744_B0842900C3352A63_u128),
        0x3AE89C3AACEE47CD_8721275248B38DDB_u128);

    assert_eq!(rust_i128_sub(3, 2), 1);
    assert_eq!(rust_i128_sub(2, 3), -1);
}

#[test]
fn test_subo() {
    assert_eq!(rust_u128_subo(3, 2), (1, false));
    assert_eq!(rust_u128_subo(2, 3), (!0, true));
    assert_eq!(rust_u128_subo(1 << 64, 1 << 63), (1 << 63, false));
    assert_eq!(rust_u128_subo(
        0xB41A6A66596687B1_3B1C743CE9F12161_u128,
        0x6019CEECA5354210_839AB51D155FF7F3_u128),
        (0x54009B79B43145A0_B781BF1FD491296E_u128, false));
    assert_eq!(rust_u128_subo(
        0x2AC60FFECA228F12_37A550530BE8B83E_u128,
        0xEFDD73C41D344744_B0842900C3352A63_u128),
        (0x3AE89C3AACEE47CD_8721275248B38DDB_u128, true));

    assert_eq!(rust_i128_subo(3, 2), (1, false));
    assert_eq!(rust_i128_subo(2, 3), (-1, false));
    assert_eq!(rust_i128_subo(1 << 64, 1 << 63), (1 << 63, false));
    assert_eq!(rust_i128_subo(
        -0x4BE59599A699784E_C4E38BC3160EDE9F_i128,
        0x6019CEECA5354210_839AB51D155FF7F3_i128),
        (0x54009B79B43145A0_B781BF1FD491296E_i128, true));
    assert_eq!(rust_i128_subo(
        0x2AC60FFECA228F12_37A550530BE8B83E_i128,
        -0x10228C3BE2CBB8BB_4F7BD6FF3CCAD59D_i128),
        (0x3AE89C3AACEE47CD_8721275248B38DDB_i128, false));
    assert_eq!(rust_i128_subo(
        0x4BE59599A699784E_C4E38BC3160EDE9F_i128,
        -0x6019CEECA5354210_839AB51D155FF7F3_i128),
        (-0x54009B79B43145A0_B781BF1FD491296E_i128, true));
    assert_eq!(rust_i128_subo(
        -0x2AC60FFECA228F12_37A550530BE8B83E_i128,
        0x10228C3BE2CBB8BB_4F7BD6FF3CCAD59D_i128),
        (-0x3AE89C3AACEE47CD_8721275248B38DDB_i128, false));
}