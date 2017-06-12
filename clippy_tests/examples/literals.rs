#![feature(plugin)]
#![plugin(clippy)]
#![warn(mixed_case_hex_literals)]
#![warn(unseparated_literal_suffix)]
#![warn(zero_prefixed_literal)]
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
    let fail_multi_zero = 000123usize;

    let ok6 = 1234_i32;
    let ok7 = 1234_f32;
    let ok8 = 1234_isize;
    let fail3 = 1234i32;
    let fail4 = 1234u32;
    let fail5 = 1234isize;
    let fail6 = 1234usize;
    let fail7 = 1.5f32;

    let ok9 = 0;
    let ok10 = 0_i64;
    let fail8 = 0123;

    let ok11 = 0o123;
    let ok12 = 0b101010;
}
