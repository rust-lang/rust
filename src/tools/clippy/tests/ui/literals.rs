// does not test any rustfixable lints

#![warn(clippy::mixed_case_hex_literals)]
#![warn(clippy::zero_prefixed_literal)]
#![warn(clippy::unseparated_literal_suffix)]
#![warn(clippy::separated_literal_suffix)]
#![allow(dead_code)]

fn main() {
    let ok1 = 0xABCD;
    let ok3 = 0xab_cd;
    let ok4 = 0xab_cd_i32;
    let ok5 = 0xAB_CD_u32;
    let ok5 = 0xAB_CD_isize;
    let fail1 = 0xabCD;
    let fail2 = 0xabCD_u32;
    let fail2 = 0xabCD_isize;
    let fail_multi_zero = 000_123usize;

    let ok9 = 0;
    let ok10 = 0_i64;
    let fail8 = 0123;

    let ok11 = 0o123;
    let ok12 = 0b10_1010;

    let ok13 = 0xab_abcd;
    let ok14 = 0xBAFE_BAFE;
    let ok15 = 0xab_cabc_abca_bcab_cabc;
    let ok16 = 0xFE_BAFE_ABAB_ABCD;
    let ok17 = 0x123_4567_8901_usize;
    let ok18 = 0xF;

    let fail19 = 12_3456_21;
    let fail22 = 3__4___23;
    let fail23 = 3__16___23;

    let fail24 = 0xAB_ABC_AB;
    let fail25 = 0b01_100_101;
    let ok26 = 0x6_A0_BF;
    let ok27 = 0b1_0010_0101;
}
