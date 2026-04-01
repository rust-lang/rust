//@no-rustfix: overlapping suggestions
// does not test any rustfixable lints

#![warn(clippy::mixed_case_hex_literals)]
#![warn(clippy::zero_prefixed_literal)]
#![warn(clippy::unseparated_literal_suffix)]
#![warn(clippy::separated_literal_suffix)]
#![allow(dead_code, overflowing_literals)]

fn main() {
    let ok1 = 0xABCD;
    let ok3 = 0xab_cd;
    let ok4 = 0xab_cd_i32;
    //~^ separated_literal_suffix

    let ok5 = 0xAB_CD_u32;
    //~^ separated_literal_suffix

    let ok5 = 0xAB_CD_isize;
    //~^ separated_literal_suffix

    let fail1 = 0xabCD;
    //~^ mixed_case_hex_literals

    let fail2 = 0xabCD_u32;
    //~^ separated_literal_suffix
    //~| mixed_case_hex_literals

    let fail2 = 0xabCD_isize;
    //~^ separated_literal_suffix
    //~| mixed_case_hex_literals

    let fail2 = 0xab_CD_isize;
    //~^ separated_literal_suffix
    //~| mixed_case_hex_literals

    let fail_multi_zero = 000_123usize;
    //~^ unseparated_literal_suffix
    //~| zero_prefixed_literal

    let ok9 = 0;
    let ok10 = 0_i64;
    //~^ separated_literal_suffix

    let fail8 = 0123;
    //~^ zero_prefixed_literal

    let ok11 = 0o123;
    let ok12 = 0b10_1010;

    let ok13 = 0xab_abcd;
    let ok14 = 0xBAFE_BAFE;
    let ok15 = 0xab_cabc_abca_bcab_cabc;
    let ok16 = 0xFE_BAFE_ABAB_ABCD;
    let ok17 = 0x123_4567_8901_usize;
    //~^ separated_literal_suffix

    let ok18 = 0xF;

    let fail19 = 12_3456_21;
    //~^ inconsistent_digit_grouping

    let fail22 = 3__4___23;
    //~^ inconsistent_digit_grouping

    let fail23 = 3__16___23;
    //~^ inconsistent_digit_grouping

    let fail24 = 0xAB_ABC_AB;
    //~^ unusual_byte_groupings

    let fail25 = 0b01_100_101;
    let ok26 = 0x6_A0_BF;
    let ok27 = 0b1_0010_0101;
}

fn issue9651() {
    // lint but octal form is not possible here
    let _ = 08;
    //~^ zero_prefixed_literal

    let _ = 09;
    //~^ zero_prefixed_literal

    let _ = 089;
    //~^ zero_prefixed_literal
}
