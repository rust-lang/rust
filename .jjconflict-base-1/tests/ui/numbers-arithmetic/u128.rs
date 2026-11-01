//@ run-pass

#![feature(test)]

extern crate test;
use test::black_box as b;

fn main() {
    let x: u128 = 0xFFFF_FFFF_FFFF_FFFF__FFFF_FFFF_FFFF_FFFF;
    assert_eq!(0, !x);
    assert_eq!(0, !x);
    let y: u128 = 0xFFFF_FFFF_FFFF_FFFF__FFFF_FFFF_FFFF_FFFE;
    assert_eq!(!1, y);
    assert_eq!(x, y | 1);
    assert_eq!(0xFAFF_0000_FF8F_0000__FFFF_0000_FFFF_FFFE,
               y &
               0xFAFF_0000_FF8F_0000__FFFF_0000_FFFF_FFFF);
    let z: u128 = 0xABCD_EF;
    assert_eq!(z * z, 0x734C_C2F2_A521);
    assert_eq!(z * z * z * z, 0x33EE_0E2A_54E2_59DA_A0E7_8E41);
    assert_eq!(z + z + z + z, 0x2AF3_7BC);
    let k: u128 = 0x1234_5678_9ABC_DEFF_EDCB_A987_6543_210;
    assert_eq!(k + k, 0x2468_ACF1_3579_BDFF_DB97_530E_CA86_420);
    assert_eq!(0, k - k);
    assert_eq!(0x1234_5678_9ABC_DEFF_EDCB_A987_5A86_421, k - z);
    assert_eq!(0x1000_0000_0000_0000_0000_0000_0000_000,
               k - 0x234_5678_9ABC_DEFF_EDCB_A987_6543_210);
    assert_eq!(0x6EF5_DE4C_D3BC_2AAA_3BB4_CC5D_D6EE_8, k / 42);
    assert_eq!(0, k % 42);
    assert_eq!(15, z % 42);
    assert_eq!(0x169D_A8020_CEC18, k % 0x3ACB_FE49_FF24_AC);
    assert_eq!(0x91A2_B3C4_D5E6_F7, k >> 65);
    assert_eq!(0xFDB9_7530_ECA8_6420_0000_0000_0000_0000, k << 65);
    assert!(k > z);
    assert!(y > k);
    assert!(y < x);
    assert_eq!(x as u64, !0);
    assert_eq!(z as u64, 0xABCD_EF);
    assert_eq!(k as u64, 0xFEDC_BA98_7654_3210);
    assert_eq!(k as i128, 0x1234_5678_9ABC_DEFF_EDCB_A987_6543_210);
    assert_eq!((z as f64) as u128, z);
    assert_eq!((z as f32) as u128, z);
    assert_eq!((z as f64 * 16.0) as u128, z * 16);
    assert_eq!((z as f32 * 16.0) as u128, z * 16);
    let l :u128 = 432 << 100;
    assert_eq!((l as f32) as u128, l);
    assert_eq!((l as f64) as u128, l);
    // formatting
    let j: u128 = 1 << 67;
    assert_eq!("147573952589676412928", format!("{}", j));
    assert_eq!("80000000000000000", format!("{:x}", j));
    assert_eq!("20000000000000000000000", format!("{:o}", j));
    assert_eq!("10000000000000000000000000000000000000000000000000000000000000000000",
               format!("{:b}", j));
    assert_eq!("340282366920938463463374607431768211455",
        format!("{}", u128::MAX));
    assert_eq!("147573952589676412928", format!("{:?}", j));
    // common traits
    assert_eq!(x, b(x.clone()));
    // overflow checks
    assert_eq!((z).checked_mul(z), Some(0x734C_C2F2_A521));
    assert_eq!((k).checked_mul(k), None);
    let l: u128 = b(u128::MAX - 10);
    let o: u128 = b(17);
    assert_eq!(l.checked_add(b(11)), None);
    assert_eq!(l.checked_sub(l), Some(0));
    assert_eq!(o.checked_sub(b(18)), None);
    assert_eq!(b(1u128).checked_shl(b(127)), Some(1 << 127));
    assert_eq!(o.checked_shl(b(128)), None);

    // Test cases for all udivmodti4 branches.
    // case "0X/0X"
    assert_eq!(b(0x69545bd57727c050_u128) /
               b(0x3283527a3350d88c_u128),
               2u128);
    // case "0X/KX"
    assert_eq!(b(0x0_8003c9c50b473ae6_u128) /
               b(0x1_283e8838c30fa8f4_u128),
               0u128);
    // case "K0/K0"
    assert_eq!(b(0xc43f42a207978720_u128 << 64) /
               b(0x098e62b74c23cf1a_u128 << 64),
               20u128);
    // case "KK/K0" for power-of-two D.
    assert_eq!(b(0xa9008fb6c9d81e42_0e25730562a601c8_u128) /
               b(1u128 << 120),
               169u128);
    // case "KK/K0" with N >= D (https://github.com/rust-lang/rust/issues/41228).
    assert_eq!(b(0xe4d26e59f0640328_06da5b06efe83a41_u128) /
               b(0x330fcb030ea4447c_u128 << 64),
               4u128);
    assert_eq!(b(3u128 << 64 | 1) /
               b(3u128 << 64),
               1u128);
    // case "KK/K0" with N < D.
    assert_eq!(b(0x6655c9fb66ca2884_e2d1dfd470158c62_u128) /
               b(0xb35b667cab7e355b_u128 << 64),
               0u128);
    // case "KX/0K" for power-of-two D.
    assert_eq!(b(0x3e49dd84feb2df59_7b2f97d93a253969_u128) /
               b(1u128 << 4),
               0x03e49dd84feb2df5_97b2f97d93a25396_u128);
    // case "KX/0K" in general.
    assert_eq!(b(0x299692b3a1dae5bd_6162e6f489d2620e_u128) /
               b(0x900b6f027571d6f7_u128),
               0x49e95f54b0442578_u128);
    // case "KX/KK" with N >= D.
    assert_eq!(b(0xc7b889180b67b07d_bc1a3c88783d35b5_u128) /
               b(0x1d7e69f53160b9e2_60074771e852f244_u128),
               6u128);
    // case "KX/KK" with N < D.
    assert_eq!(b(0x679289ac23bb334f_36144401cf882172_u128) /
               b(0x7b0b271b64865f05_f54a7b72746c062f_u128),
               0u128);

    // iter-arithmetic traits
    assert_eq!(10u128, [1u128, 2, 3, 4].iter().sum());
    assert_eq!(24u128, [1u128, 2, 3, 4].iter().product());
}
