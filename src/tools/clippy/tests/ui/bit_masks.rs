const THREE_BITS: i64 = 7;
const EVEN_MORE_REDIRECTION: i64 = THREE_BITS;

#[warn(clippy::bad_bit_mask)]
#[allow(
    clippy::ineffective_bit_mask,
    clippy::identity_op,
    clippy::no_effect,
    clippy::unnecessary_operation
)]
fn main() {
    let x = 5;

    x & 0 == 0;
    //~^ ERROR: &-masking with zero
    //~| NOTE: `-D clippy::bad-bit-mask` implied by `-D warnings`
    //~| ERROR: this operation will always return zero. This is likely not the intended ou
    //~| NOTE: `#[deny(clippy::erasing_op)]` on by default
    x & 1 == 1; //ok, distinguishes bit 0
    x & 1 == 0; //ok, compared with zero
    x & 2 == 1;
    //~^ ERROR: incompatible bit mask: `_ & 2` can never be equal to `1`
    x | 0 == 0; //ok, equals x == 0 (maybe warn?)
    x | 1 == 3; //ok, equals x == 2 || x == 3
    x | 3 == 3; //ok, equals x <= 3
    x | 3 == 2;
    //~^ ERROR: incompatible bit mask: `_ | 3` can never be equal to `2`

    x & 1 > 1;
    //~^ ERROR: incompatible bit mask: `_ & 1` will never be higher than `1`
    x & 2 > 1; // ok, distinguishes x & 2 == 2 from x & 2 == 0
    x & 2 < 1; // ok, distinguishes x & 2 == 2 from x & 2 == 0
    x | 1 > 1; // ok (if a bit silly), equals x > 1
    x | 2 > 1;
    //~^ ERROR: incompatible bit mask: `_ | 2` will always be higher than `1`
    x | 2 <= 2; // ok (if a bit silly), equals x <= 2

    x & 192 == 128; // ok, tests for bit 7 and not bit 6
    x & 0xffc0 == 0xfe80; // ok

    // this also now works with constants
    x & THREE_BITS == 8;
    //~^ ERROR: incompatible bit mask: `_ & 7` can never be equal to `8`
    x | EVEN_MORE_REDIRECTION < 7;
    //~^ ERROR: incompatible bit mask: `_ | 7` will never be lower than `7`

    0 & x == 0;
    //~^ ERROR: &-masking with zero
    //~| ERROR: this operation will always return zero. This is likely not the intended ou
    1 | x > 1;

    // and should now also match uncommon usage
    1 < 2 | x;
    //~^ ERROR: incompatible bit mask: `_ | 2` will always be higher than `1`
    2 == 3 | x;
    //~^ ERROR: incompatible bit mask: `_ | 3` can never be equal to `2`
    1 == x & 2;
    //~^ ERROR: incompatible bit mask: `_ & 2` can never be equal to `1`

    x | 1 > 2; // no error, because we allowed ineffective bit masks
    ineffective();
}

#[warn(clippy::ineffective_bit_mask)]
#[allow(clippy::bad_bit_mask, clippy::no_effect, clippy::unnecessary_operation)]
fn ineffective() {
    let x = 5;

    x | 1 > 3;
    //~^ ERROR: ineffective bit mask: `x | 1` compared to `3`, is the same as x compared d
    //~| NOTE: `-D clippy::ineffective-bit-mask` implied by `-D warnings`
    x | 1 < 4;
    //~^ ERROR: ineffective bit mask: `x | 1` compared to `4`, is the same as x compared d
    x | 1 <= 3;
    //~^ ERROR: ineffective bit mask: `x | 1` compared to `3`, is the same as x compared d
    x | 1 >= 8;
    //~^ ERROR: ineffective bit mask: `x | 1` compared to `8`, is the same as x compared d

    x | 1 > 2; // not an error (yet), better written as x >= 2
    x | 1 >= 7; // not an error (yet), better written as x >= 6
    x | 3 > 4; // not an error (yet), better written as x >= 4
    x | 4 <= 19;
}
