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
    //~^ ERROR: integer type suffix should not be separated by an underscore
    //~| NOTE: `-D clippy::separated-literal-suffix` implied by `-D warnings`
    let ok5 = 0xAB_CD_u32;
    //~^ ERROR: integer type suffix should not be separated by an underscore
    let ok5 = 0xAB_CD_isize;
    //~^ ERROR: integer type suffix should not be separated by an underscore
    let fail1 = 0xabCD;
    //~^ ERROR: inconsistent casing in hexadecimal literal
    //~| NOTE: `-D clippy::mixed-case-hex-literals` implied by `-D warnings`
    let fail2 = 0xabCD_u32;
    //~^ ERROR: integer type suffix should not be separated by an underscore
    //~| ERROR: inconsistent casing in hexadecimal literal
    let fail2 = 0xabCD_isize;
    //~^ ERROR: integer type suffix should not be separated by an underscore
    //~| ERROR: inconsistent casing in hexadecimal literal
    let fail_multi_zero = 000_123usize;
    //~^ ERROR: integer type suffix should be separated by an underscore
    //~| NOTE: `-D clippy::unseparated-literal-suffix` implied by `-D warnings`
    //~| ERROR: this is a decimal constant
    //~| NOTE: `-D clippy::zero-prefixed-literal` implied by `-D warnings`

    let ok9 = 0;
    let ok10 = 0_i64;
    //~^ ERROR: integer type suffix should not be separated by an underscore
    let fail8 = 0123;
    //~^ ERROR: this is a decimal constant

    let ok11 = 0o123;
    let ok12 = 0b10_1010;

    let ok13 = 0xab_abcd;
    let ok14 = 0xBAFE_BAFE;
    let ok15 = 0xab_cabc_abca_bcab_cabc;
    let ok16 = 0xFE_BAFE_ABAB_ABCD;
    let ok17 = 0x123_4567_8901_usize;
    //~^ ERROR: integer type suffix should not be separated by an underscore
    let ok18 = 0xF;

    let fail19 = 12_3456_21;
    //~^ ERROR: digits grouped inconsistently by underscores
    //~| NOTE: `-D clippy::inconsistent-digit-grouping` implied by `-D warnings`
    let fail22 = 3__4___23;
    //~^ ERROR: digits grouped inconsistently by underscores
    let fail23 = 3__16___23;
    //~^ ERROR: digits grouped inconsistently by underscores

    let fail24 = 0xAB_ABC_AB;
    //~^ ERROR: digits of hex, binary or octal literal not in groups of equal size
    //~| NOTE: `-D clippy::unusual-byte-groupings` implied by `-D warnings`
    let fail25 = 0b01_100_101;
    let ok26 = 0x6_A0_BF;
    let ok27 = 0b1_0010_0101;
}

fn issue9651() {
    // lint but octal form is not possible here
    let _ = 08;
    //~^ ERROR: this is a decimal constant
    let _ = 09;
    //~^ ERROR: this is a decimal constant
    let _ = 089;
    //~^ ERROR: this is a decimal constant
}
