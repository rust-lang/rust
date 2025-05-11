#![allow(unused)]
#![warn(clippy::implicit_saturating_add)]

fn main() {
    let mut u_8: u8 = 255;
    let mut u_16: u16 = 500;
    let mut u_32: u32 = 7000;
    let mut u_64: u64 = 7000;
    let mut i_8: i8 = 30;
    let mut i_16: i16 = 500;
    let mut i_32: i32 = 7000;
    let mut i_64: i64 = 7000;

    if i_8 < 42 {
        i_8 += 1;
    }
    if i_8 != 42 {
        i_8 += 1;
    }

    if u_8 != u8::MAX {
        //~^ implicit_saturating_add
        u_8 += 1;
    }

    if u_8 < u8::MAX {
        //~^ implicit_saturating_add
        u_8 += 1;
    }

    if u_8 < 15 {
        u_8 += 1;
    }

    if u_16 != u16::MAX {
        //~^ implicit_saturating_add
        u_16 += 1;
    }

    if u_16 < u16::MAX {
        //~^ implicit_saturating_add
        u_16 += 1;
    }

    if u16::MAX > u_16 {
        //~^ implicit_saturating_add
        u_16 += 1;
    }

    if u_32 != u32::MAX {
        //~^ implicit_saturating_add
        u_32 += 1;
    }

    if u_32 < u32::MAX {
        //~^ implicit_saturating_add
        u_32 += 1;
    }

    if u32::MAX > u_32 {
        //~^ implicit_saturating_add
        u_32 += 1;
    }

    if u_64 != u64::MAX {
        //~^ implicit_saturating_add
        u_64 += 1;
    }

    if u_64 < u64::MAX {
        //~^ implicit_saturating_add
        u_64 += 1;
    }

    if u64::MAX > u_64 {
        //~^ implicit_saturating_add
        u_64 += 1;
    }

    if i_8 != i8::MAX {
        //~^ implicit_saturating_add
        i_8 += 1;
    }

    if i_8 < i8::MAX {
        //~^ implicit_saturating_add
        i_8 += 1;
    }

    if i8::MAX > i_8 {
        //~^ implicit_saturating_add
        i_8 += 1;
    }

    if i_16 != i16::MAX {
        //~^ implicit_saturating_add
        i_16 += 1;
    }

    if i_16 < i16::MAX {
        //~^ implicit_saturating_add
        i_16 += 1;
    }

    if i16::MAX > i_16 {
        //~^ implicit_saturating_add
        i_16 += 1;
    }

    if i_32 != i32::MAX {
        //~^ implicit_saturating_add
        i_32 += 1;
    }

    if i_32 < i32::MAX {
        //~^ implicit_saturating_add
        i_32 += 1;
    }

    if i32::MAX > i_32 {
        //~^ implicit_saturating_add
        i_32 += 1;
    }

    if i_64 != i64::MAX {
        //~^ implicit_saturating_add
        i_64 += 1;
    }

    if i_64 < i64::MAX {
        //~^ implicit_saturating_add
        i_64 += 1;
    }

    if i64::MAX > i_64 {
        //~^ implicit_saturating_add
        i_64 += 1;
    }

    if i_64 < 42 {
        i_64 += 1;
    }

    if 42 > i_64 {
        i_64 += 1;
    }

    let a = 12;
    let mut b = 8;

    if a < u8::MAX {
        b += 1;
    }

    if u8::MAX > a {
        b += 1;
    }

    if u_32 < u32::MAX {
        u_32 += 1;
    } else {
        println!("don't lint this");
    }

    if u_32 < u32::MAX {
        println!("don't lint this");
        u_32 += 1;
    }

    if u_32 < 42 {
        println!("brace yourself!");
    } else if u_32 < u32::MAX {
        //~^ implicit_saturating_add
        u_32 += 1;
    }
}
