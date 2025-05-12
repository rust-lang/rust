//@ check-pass

#![warn(unused)]

// This expect attribute should catch all lint triggers
#[expect(unused_variables)]
fn check_multiple_lints_1() {
    let value_i = 0xff00ff;
    let value_ii = 0xff00ff;
    let value_iii = 0xff00ff;
    let value_iiii = 0xff00ff;
    let value_iiiii = 0xff00ff;
}

// This expect attribute should catch all lint triggers
#[expect(unused_mut)]
fn check_multiple_lints_2() {
    let mut a = 0xa;
    let mut b = 0xb;
    let mut c = 0xc;
    println!("The ABC goes as: {:#x} {:#x} {:#x}", a, b, c);
}

// This expect attribute should catch all lint triggers
#[expect(while_true)]
fn check_multiple_lints_3() {
    // `while_true` is an early lint
    while true {}

    while true {}

    while true {}

    while true {}

    while true {}
}

fn main() {
    check_multiple_lints_1();
    check_multiple_lints_2();
    check_multiple_lints_3();
}
