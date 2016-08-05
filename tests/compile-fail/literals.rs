#![feature(plugin)]
#![plugin(clippy)]
#![deny(mixed_case_hex_literals)]
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
}
