//@ check-pass
#![warn(overflowing_literals)]

fn main() {
    let error = 255i8; //~WARNING literal out of range for `i8`
    //~^ HELP consider using the type `u8` instead

    let ok = 0b1000_0001; // should be ok -> i32
    let ok = 0b0111_1111i8; // should be ok -> 127i8

    let fail = 0b1000_0001i8; //~WARNING literal out of range for `i8`
    //~^ HELP consider using the type `u8` instead
    //~| HELP consider using the type `u8` for the literal and cast it to `i8`

    let fail = 0x8000_0000_0000_0000i64; //~WARNING literal out of range for `i64`
    //~^ HELP consider using the type `u64` instead
    //~| HELP consider using the type `u64` for the literal and cast it to `i64`

    let fail = 0x1_FFFF_FFFFu32; //~WARNING literal out of range for `u32`
    //~^ HELP consider using the type `u64` instead

    let fail: i128 = 0x8000_0000_0000_0000_0000_0000_0000_0000;
    //~^ WARNING literal out of range for `i128`
    //~| HELP consider using the type `u128` instead
    //~| HELP consider using the type `u128` for the literal and cast it to `i128`

    let fail = 0x8000_0000_0000_0000_0000_0000_0000_0000;
    //~^ WARNING literal out of range for `i32`
    //~| HELP consider using the type `u128` instead

    let fail = -0x8000_0000_0000_0000_0000_0000_0000_0000; // issue #131849
    //~^ WARNING literal out of range for `i32`
    //~| HELP consider using the type `i128` instead

    let fail = -0x8000_0000_0000_0000_0000_0000_0000_0001i128;
    //~^ WARNING literal out of range for `i128`

    let fail = 340282366920938463463374607431768211455i8;
    //~^ WARNING literal out of range for `i8`
    //~| HELP consider using the type `u128` instead

    let fail = 0x8FFF_FFFF_FFFF_FFFE; //~WARNING literal out of range for `i32`
    //~| HELP consider using the type `u64` instead
    //~| HELP consider using the type `u64` for the literal and cast it to `i32`

    let fail = -0b1111_1111i8; //~WARNING literal out of range for `i8`
    //~| HELP consider using the type `i16` instead

    let fail = 0x8000_0000_0000_0000_0000_0000_FFFF_FFFE; //~WARNING literal out of range for `i32`
    //~| HELP consider using the type `u128` instead
    //~| HELP consider using the type `u128` for the literal and cast it to `i32`
}
