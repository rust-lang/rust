#![feature(plugin)]
#![plugin(clippy)]
#![deny(mixed_case_hex_literals)]
#![deny(unseparated_literal_suffix)]
#![deny(zero_prefixed_literal)]
#![allow(dead_code)]

fn main() {
    let ok1 = 0xABCD;
    let ok3 = 0xab_cd;
    let ok4 = 0xab_cd_i32;
    let ok5 = 0xAB_CD_u32;
    let ok5 = 0xAB_CD_isize;
    let fail1 = 0xabCD;       //~ERROR inconsistent casing in hexadecimal literal
    let fail2 = 0xabCD_u32;   //~ERROR inconsistent casing in hexadecimal literal
    let fail2 = 0xabCD_isize; //~ERROR inconsistent casing in hexadecimal literal

    let ok6 = 1234_i32;
    let ok7 = 1234_f32;
    let ok8 = 1234_isize;
    let fail3 = 1234i32;      //~ERROR integer type suffix should be separated
    let fail4 = 1234u32;      //~ERROR integer type suffix should be separated
    let fail5 = 1234isize;    //~ERROR integer type suffix should be separated
    let fail6 = 1234usize;    //~ERROR integer type suffix should be separated
    let fail7 = 1.5f32;       //~ERROR float type suffix should be separated

    let ok9 = 0;
    let ok10 = 0_i64;
    let fail8 = 0123;
    //~^ERROR decimal constant
    //~|HELP remove the `0`
    //~|SUGGESTION = 123;
    //~|HELP use `0o`
    //~|SUGGESTION = 0o123;

    let ok11 = 0o123;
    let ok12 = 0b101010;
}
