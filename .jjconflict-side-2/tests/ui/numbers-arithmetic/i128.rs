//@ run-pass
#![allow(overflowing_literals)]

#![feature(test)]

extern crate test;
use test::black_box as b;

fn main() {
    let x: i128 = -1;
    assert_eq!(0, !x);
    let y: i128 = -2;
    assert_eq!(!1, y);
    let z: i128 = 0xABCD_EF;
    assert_eq!(z * z, 0x734C_C2F2_A521);
    assert_eq!(z * z * z * z, 0x33EE_0E2A_54E2_59DA_A0E7_8E41);
    assert_eq!(-z * -z, 0x734C_C2F2_A521);
    assert_eq!(-z * -z * -z * -z, 0x33EE_0E2A_54E2_59DA_A0E7_8E41);
    assert_eq!(-z + -z + -z + -z, -0x2AF3_7BC);
    let k: i128 = -0x1234_5678_9ABC_DEFF_EDCB_A987_6543_210;
    assert_eq!(k + k, -0x2468_ACF1_3579_BDFF_DB97_530E_CA86_420);
    assert_eq!(0, k - k);
    assert_eq!(-0x1234_5678_9ABC_DEFF_EDCB_A987_5A86_421, k + z);
    assert_eq!(-0x1000_0000_0000_0000_0000_0000_0000_000,
               k + 0x234_5678_9ABC_DEFF_EDCB_A987_6543_210);
    assert_eq!(-0x6EF5_DE4C_D3BC_2AAA_3BB4_CC5D_D6EE_8, k / 42);
    assert_eq!(-k, k / -1);
    assert_eq!(-0x91A2_B3C4_D5E6_F8, k >> 65);
    assert_eq!(-0xFDB9_7530_ECA8_6420_0000_0000_0000_0000, k << 65);
    assert!(k < z);
    assert!(y > k);
    assert!(y < x);
    assert_eq!(x as i64, -1);
    assert_eq!(z as i64, 0xABCD_EF);
    assert_eq!(k as i64, -0xFEDC_BA98_7654_3210);
    assert_eq!(k as u128, 0xFEDC_BA98_7654_3210_0123_4567_89AB_CDF0);
    assert_eq!(-k as u128, 0x1234_5678_9ABC_DEFF_EDCB_A987_6543_210);
    assert_eq!((-z as f64) as i128, -z);
    assert_eq!((-z as f32) as i128, -z);
    assert_eq!((-z as f64 * 16.0) as i128, -z * 16);
    assert_eq!((-z as f32 * 16.0) as i128, -z * 16);
    // Same stuff as above, but blackboxed, to force use of intrinsics
    let x: i128 = b(-1);
    assert_eq!(0, !x);
    let y: i128 = b(-2);
    assert_eq!(!1, y);
    let z: i128 = b(0xABCD_EF);
    assert_eq!(z * z, 0x734C_C2F2_A521);
    assert_eq!(z * z * z * z, 0x33EE_0E2A_54E2_59DA_A0E7_8E41);
    assert_eq!(-z * -z, 0x734C_C2F2_A521);
    assert_eq!(-z * -z * -z * -z, 0x33EE_0E2A_54E2_59DA_A0E7_8E41);
    assert_eq!(-z + -z + -z + -z, -0x2AF3_7BC);
    let k: i128 = b(-0x1234_5678_9ABC_DEFF_EDCB_A987_6543_210);
    assert_eq!(k + k, -0x2468_ACF1_3579_BDFF_DB97_530E_CA86_420);
    assert_eq!(0, k - k);
    assert_eq!(-0x1234_5678_9ABC_DEFF_EDCB_A987_5A86_421, k + z);
    assert_eq!(-0x1000_0000_0000_0000_0000_0000_0000_000,
               k + 0x234_5678_9ABC_DEFF_EDCB_A987_6543_210);
    assert_eq!(-0x6EF5_DE4C_D3BC_2AAA_3BB4_CC5D_D6EE_8, k / 42);
    assert_eq!(-k, k / -1);
    assert_eq!(-0x91A2_B3C4_D5E6_F8, k >> 65);
    assert_eq!(-0xFDB9_7530_ECA8_6420_0000_0000_0000_0000, k << 65);
    assert!(k < z);
    assert!(y > k);
    assert!(y < x);
    assert_eq!(x as i64, -1);
    assert_eq!(z as i64, 0xABCD_EF);
    assert_eq!(k as i64, -0xFEDC_BA98_7654_3210);
    assert_eq!(k as u128, 0xFEDC_BA98_7654_3210_0123_4567_89AB_CDF0);
    assert_eq!(-k as u128, 0x1234_5678_9ABC_DEFF_EDCB_A987_6543_210);
    assert_eq!((-z as f64) as i128, -z);
    assert_eq!((-z as f32) as i128, -z);
    assert_eq!((-z as f64 * 16.0) as i128, -z * 16);
    assert_eq!((-z as f32 * 16.0) as i128, -z * 16);
    // formatting
    let j: i128 = -(1 << 67);
    assert_eq!("-147573952589676412928", format!("{}", j));
    assert_eq!("fffffffffffffff80000000000000000", format!("{:x}", j));
    assert_eq!("3777777777777777777760000000000000000000000", format!("{:o}", j));
    assert_eq!("1111111111111111111111111111111111111111111111111111111111111\
                0000000000000000000000000000000000000000000000000000000000000000000",
               format!("{:b}", j));
    assert_eq!("-147573952589676412928", format!("{:?}", j));
    // common traits
    assert_eq!(x, b(x.clone()));
    // overflow checks
    assert_eq!((-z).checked_mul(-z), Some(0x734C_C2F2_A521));
    assert_eq!((z).checked_mul(z), Some(0x734C_C2F2_A521));
    assert_eq!((k).checked_mul(k), None);
    let l: i128 = b(i128::MIN);
    let o: i128 = b(17);
    assert_eq!(l.checked_sub(b(2)), None);
    assert_eq!(l.checked_add(l), None);
    assert_eq!((-(l + 1)).checked_add(2), None);
    assert_eq!(l.checked_sub(l), Some(0));
    assert_eq!(b(1u128).checked_shl(b(127)), Some(1 << 127));
    assert_eq!(o.checked_shl(b(128)), None);

    // https://github.com/rust-lang/rust/issues/41228
    assert_eq!(b(-87559967289969187895646876466835277875_i128) /
               b(84285771033834995895337664386045050880_i128),
               -1i128);

    // iter-arithmetic traits
    assert_eq!(10i128, [1i128, 2, 3, 4].iter().sum());
    assert_eq!(24i128, [1i128, 2, 3, 4].iter().product());
}
