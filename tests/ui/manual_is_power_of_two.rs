#![warn(clippy::manual_is_power_of_two)]
#![allow(clippy::precedence)]

macro_rules! binop {
    ($a: expr, equal, $b: expr) => {
        $a == $b
    };
    ($a: expr, and, $b: expr) => {
        $a & $b
    };
    ($a: expr, minus, $b: expr) => {
        $a - $b
    };
}

fn main() {
    let a = 16_u64;

    let _ = a.count_ones() == 1;
    //~^ manual_is_power_of_two
    let _ = u64::count_ones(a) == 1;
    //~^ manual_is_power_of_two
    let _ = a & (a - 1) == 0;
    //~^ manual_is_power_of_two

    // Test different orders of expression
    let _ = 1 == a.count_ones();
    //~^ manual_is_power_of_two
    let _ = (a - 1) & a == 0;
    //~^ manual_is_power_of_two
    let _ = 0 == a & (a - 1);
    //~^ manual_is_power_of_two
    let _ = 0 == (a - 1) & a;
    //~^ manual_is_power_of_two

    let b = 4_i64;

    // is_power_of_two only works for unsigned integers
    let _ = b.count_ones() == 1;
    let _ = b & (b - 1) == 0;

    let i: i32 = 3;
    let _ = i as u32 & (i as u32 - 1) == 0;
    //~^ manual_is_power_of_two

    let _ = binop!(a.count_ones(), equal, 1);
    let _ = binop!(a, and, a - 1) == 0;
    let _ = a & binop!(a, minus, 1) == 0;
}

#[clippy::msrv = "1.31"]
const fn low_msrv(a: u32) -> bool {
    a & (a - 1) == 0
}

#[clippy::msrv = "1.32"]
const fn high_msrv(a: u32) -> bool {
    a & (a - 1) == 0
    //~^ manual_is_power_of_two
}
