#![warn(clippy::invalid_upcast_comparisons)]
#![allow(
    unused,
    clippy::eq_op,
    clippy::no_effect,
    clippy::unnecessary_operation,
    clippy::cast_lossless
)]

fn mk_value<T>() -> T {
    unimplemented!()
}

fn main() {
    let u32: u32 = mk_value();
    let u8: u8 = mk_value();
    let i32: i32 = mk_value();
    let i8: i8 = mk_value();

    // always false, since no u8 can be > 300
    (u8 as u32) > 300;
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is
    //~| NOTE: `-D clippy::invalid-upcast-comparisons` implied by `-D warnings`
    (u8 as i32) > 300;
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is
    (u8 as u32) == 300;
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is
    (u8 as i32) == 300;
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is
    300 < (u8 as u32);
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is
    300 < (u8 as i32);
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is
    300 == (u8 as u32);
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is
    300 == (u8 as i32);
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is
    // inverted of the above
    (u8 as u32) <= 300;
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is
    (u8 as i32) <= 300;
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is
    (u8 as u32) != 300;
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is
    (u8 as i32) != 300;
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is
    300 >= (u8 as u32);
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is
    300 >= (u8 as i32);
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is
    300 != (u8 as u32);
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is
    300 != (u8 as i32);
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is

    // always false, since u8 -> i32 doesn't wrap
    (u8 as i32) < 0;
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is
    -5 != (u8 as i32);
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is
    // inverted of the above
    (u8 as i32) >= 0;
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is
    -5 == (u8 as i32);
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is

    // always false, since no u8 can be 1337
    1337 == (u8 as i32);
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is
    1337 == (u8 as u32);
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is
    // inverted of the above
    1337 != (u8 as i32);
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is
    1337 != (u8 as u32);
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is

    // Those are Ok:
    (u8 as u32) > 20;
    42 == (u8 as i32);
    42 != (u8 as i32);
    42 > (u8 as i32);
    (u8 as i32) == 42;
    (u8 as i32) != 42;
    (u8 as i32) > 42;
    (u8 as i32) < 42;

    (u8 as i8) == -1;
    (u8 as i8) != -1;
    (u8 as i32) > -1;
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is
    (u8 as i32) < -1;
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is
    (u32 as i32) < -5;
    (u32 as i32) < 10;

    (i8 as u8) == 1;
    (i8 as u8) != 1;
    (i8 as u8) < 1;
    (i8 as u8) > 1;
    (i32 as u32) < 5;
    (i32 as u32) < 10;

    -5 < (u32 as i32);
    0 <= (u32 as i32);
    0 < (u32 as i32);

    -5 > (u32 as i32);
    -5 >= (u8 as i32);
    //~^ ERROR: because of the numeric bounds on `u8` prior to casting, this expression is

    -5 == (u32 as i32);
}
