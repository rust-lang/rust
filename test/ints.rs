#![crate_type = "lib"]
#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

#[miri_run]
fn ret() -> i64 {
    1
}

#[miri_run]
fn neg() -> i64 {
    -1
}

#[miri_run]
fn add() -> i64 {
    1 + 2
}

#[miri_run]
fn indirect_add() -> i64 {
    let x = 1;
    let y = 2;
    x + y
}

#[miri_run]
fn arith() -> i32 {
    3*3 + 4*4
}

#[miri_run]
fn match_int() -> i16 {
    let n = 2;
    match n {
        0 => 0,
        1 => 10,
        2 => 20,
        3 => 30,
        _ => 100,
    }
}

#[miri_run]
fn match_int_range() -> i64 {
    let n = 42;
    match n {
        0...9 => 0,
        10...19 => 1,
        20...29 => 2,
        30...39 => 3,
        40...49 => 4,
        _ => 5,
    }
}
