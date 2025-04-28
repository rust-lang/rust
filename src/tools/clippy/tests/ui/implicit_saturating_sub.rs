#![allow(unused_assignments, unused_mut, clippy::assign_op_pattern)]
#![warn(clippy::implicit_saturating_sub)]

use std::cmp::PartialEq;
use std::ops::SubAssign;
// Mock type
struct Mock;

impl PartialEq<u32> for Mock {
    fn eq(&self, _: &u32) -> bool {
        true
    }
}

impl SubAssign<u32> for Mock {
    fn sub_assign(&mut self, _: u32) {}
}

fn main() {
    // Tests for unsigned integers

    let end_8: u8 = 10;
    let start_8: u8 = 5;
    let mut u_8: u8 = end_8 - start_8;

    // Lint
    if u_8 > 0 {
        //~^ implicit_saturating_sub
        u_8 = u_8 - 1;
    }

    match end_8 {
        10 => {
            // Lint
            if u_8 > 0 {
                //~^ implicit_saturating_sub
                u_8 -= 1;
            }
        },
        11 => u_8 += 1,
        _ => u_8 = 0,
    }

    let end_16: u16 = 40;
    let start_16: u16 = 35;

    let mut u_16: u16 = end_16 - start_16;

    // Lint
    if u_16 > 0 {
        //~^ implicit_saturating_sub
        u_16 -= 1;
    }

    let mut end_32: u32 = 7010;
    let mut start_32: u32 = 7000;

    let mut u_32: u32 = end_32 - start_32;

    // Lint
    if u_32 != 0 {
        //~^ implicit_saturating_sub
        u_32 -= 1;
    }

    // No Lint
    if u_32 > 0 {
        u_16 += 1;
    }

    // No Lint
    if u_32 != 0 {
        end_32 -= 1;
        start_32 += 1;
    }

    let mut end_64: u64 = 75001;
    let mut start_64: u64 = 75000;

    let mut u_64: u64 = end_64 - start_64;

    // Lint
    if u_64 > 0 {
        //~^ implicit_saturating_sub
        u_64 -= 1;
    }

    // Lint
    if 0 < u_64 {
        //~^ implicit_saturating_sub
        u_64 -= 1;
    }

    // Lint
    if 0 != u_64 {
        //~^ implicit_saturating_sub
        u_64 -= 1;
    }

    // No Lint
    if u_64 >= 1 {
        u_64 -= 1;
    }

    // No Lint
    if u_64 > 0 {
        end_64 -= 1;
    }

    // Tests for usize
    let end_usize: usize = 8054;
    let start_usize: usize = 8050;

    let mut u_usize: usize = end_usize - start_usize;

    // Lint
    if u_usize > 0 {
        //~^ implicit_saturating_sub
        u_usize -= 1;
    }

    // Tests for signed integers

    let endi_8: i8 = 10;
    let starti_8: i8 = 50;

    let mut i_8: i8 = endi_8 - starti_8;

    // Lint
    if i_8 > i8::MIN {
        //~^ implicit_saturating_sub
        i_8 -= 1;
    }

    // Lint
    if i_8 > i8::MIN {
        //~^ implicit_saturating_sub
        i_8 -= 1;
    }

    // Lint
    if i_8 != i8::MIN {
        //~^ implicit_saturating_sub
        i_8 -= 1;
    }

    // Lint
    if i_8 != i8::MIN {
        //~^ implicit_saturating_sub
        i_8 -= 1;
    }

    let endi_16: i16 = 45;
    let starti_16: i16 = 44;

    let mut i_16: i16 = endi_16 - starti_16;

    // Lint
    if i_16 > i16::MIN {
        //~^ implicit_saturating_sub
        i_16 -= 1;
    }

    // Lint
    if i_16 > i16::MIN {
        //~^ implicit_saturating_sub
        i_16 -= 1;
    }

    // Lint
    if i_16 != i16::MIN {
        //~^ implicit_saturating_sub
        i_16 -= 1;
    }

    // Lint
    if i_16 != i16::MIN {
        //~^ implicit_saturating_sub
        i_16 -= 1;
    }

    let endi_32: i32 = 45;
    let starti_32: i32 = 44;

    let mut i_32: i32 = endi_32 - starti_32;

    // Lint
    if i_32 > i32::MIN {
        //~^ implicit_saturating_sub
        i_32 -= 1;
    }

    // Lint
    if i_32 > i32::MIN {
        //~^ implicit_saturating_sub
        i_32 -= 1;
    }

    // Lint
    if i_32 != i32::MIN {
        //~^ implicit_saturating_sub
        i_32 -= 1;
    }

    // Lint
    if i_32 != i32::MIN {
        //~^ implicit_saturating_sub
        i_32 -= 1;
    }

    let endi_64: i64 = 45;
    let starti_64: i64 = 44;

    let mut i_64: i64 = endi_64 - starti_64;

    // Lint
    if i64::MIN < i_64 {
        //~^ implicit_saturating_sub
        i_64 -= 1;
    }

    // Lint
    if i64::MIN != i_64 {
        //~^ implicit_saturating_sub
        i_64 -= 1;
    }

    // Lint
    if i64::MIN < i_64 {
        //~^ implicit_saturating_sub
        i_64 -= 1;
    }

    // No Lint
    if i_64 > 0 {
        i_64 -= 1;
    }

    // No Lint
    if i_64 != 0 {
        i_64 -= 1;
    }

    // issue #7831
    // No Lint
    if u_32 > 0 {
        u_32 -= 1;
    } else {
        println!("side effect");
    }

    // Extended tests
    let mut m = Mock;
    let mut u_32 = 3000;
    let a = 200;
    let mut b = 8;

    if m != 0 {
        m -= 1;
    }

    if a > 0 {
        b -= 1;
    }

    if 0 > a {
        b -= 1;
    }

    if u_32 > 0 {
        u_32 -= 1;
    } else {
        println!("don't lint this");
    }

    if u_32 > 0 {
        println!("don't lint this");
        u_32 -= 1;
    }

    if u_32 > 42 {
        println!("brace yourself!");
    } else if u_32 > 0 {
        u_32 -= 1;
    }

    let result = if a < b {
        println!("we shouldn't remove this");
        0
    } else {
        a - b
    };
}

fn regression_13524(a: usize, b: usize, c: bool) -> usize {
    if c {
        123
    } else if a >= b {
        //~^ implicit_saturating_sub
        0
    } else {
        b - a
    }
}

fn with_side_effect(a: u64) -> u64 {
    println!("a = {a}");
    a
}

fn arbitrary_expression() {
    let (a, b) = (15u64, 20u64);

    let _ = if a * 2 > b { a * 2 - b } else { 0 };
    //~^ implicit_saturating_sub

    let _ = if a > b * 2 { a - b * 2 } else { 0 };
    //~^ implicit_saturating_sub

    let _ = if a < b * 2 { 0 } else { a - b * 2 };
    //~^ implicit_saturating_sub

    let _ = if with_side_effect(a) > a {
        with_side_effect(a) - a
    } else {
        0
    };
}
