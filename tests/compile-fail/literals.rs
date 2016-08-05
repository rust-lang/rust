#![feature(plugin)]
#![plugin(clippy)]
#![deny(mixed_case_hex_literals)]
#![deny(unseparated_literal_suffix)]
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
}
