// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
#![allow(non_upper_case_globals)]

fn ret() -> u32 {
    static x: u32 = 10;
    x & if true { 10u32 } else { 20u32 } & x
}

fn ret2() -> &'static u32 {
    static x: u32 = 10;
    if true { 10u32; } else { 20u32; }
    &x
}

fn main() {}
