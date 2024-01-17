//@no-rustfix

#![feature(repr128)]
#![feature(isqrt)]
#![allow(incomplete_features)]
#![warn(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
#![allow(clippy::cast_abs_to_unsigned, clippy::no_effect, clippy::unnecessary_operation)]

fn main() {
    // Test clippy::cast_precision_loss
    let x0 = 1i32;
    x0 as f32;
    //~^ ERROR: casting `i32` to `f32` causes a loss of precision (`i32` is 32 bits wide,
    //~| NOTE: `-D clippy::cast-precision-loss` implied by `-D warnings`
    let x1 = 1i64;
    x1 as f32;
    //~^ ERROR: casting `i64` to `f32` causes a loss of precision (`i64` is 64 bits wide,
    x1 as f64;
    //~^ ERROR: casting `i64` to `f64` causes a loss of precision (`i64` is 64 bits wide,
    let x2 = 1u32;
    x2 as f32;
    //~^ ERROR: casting `u32` to `f32` causes a loss of precision (`u32` is 32 bits wide,
    let x3 = 1u64;
    x3 as f32;
    //~^ ERROR: casting `u64` to `f32` causes a loss of precision (`u64` is 64 bits wide,
    x3 as f64;
    //~^ ERROR: casting `u64` to `f64` causes a loss of precision (`u64` is 64 bits wide,
    // Test clippy::cast_possible_truncation
    1f32 as i32;
    //~^ ERROR: casting `f32` to `i32` may truncate the value
    1f32 as u32;
    //~^ ERROR: casting `f32` to `u32` may truncate the value
    //~| ERROR: casting `f32` to `u32` may lose the sign of the value
    //~| NOTE: `-D clippy::cast-sign-loss` implied by `-D warnings`
    1f64 as f32;
    //~^ ERROR: casting `f64` to `f32` may truncate the value
    1i32 as i8;
    //~^ ERROR: casting `i32` to `i8` may truncate the value
    1i32 as u8;
    //~^ ERROR: casting `i32` to `u8` may truncate the value
    1f64 as isize;
    //~^ ERROR: casting `f64` to `isize` may truncate the value
    1f64 as usize;
    //~^ ERROR: casting `f64` to `usize` may truncate the value
    //~| ERROR: casting `f64` to `usize` may lose the sign of the value
    1f32 as u32 as u16;
    //~^ ERROR: casting `u32` to `u16` may truncate the value
    //~| ERROR: casting `f32` to `u32` may truncate the value
    //~| ERROR: casting `f32` to `u32` may lose the sign of the value
    {
        let _x: i8 = 1i32 as _;
        //~^ ERROR: casting `i32` to `i8` may truncate the value
        1f32 as i32;
        //~^ ERROR: casting `f32` to `i32` may truncate the value
        1f64 as i32;
        //~^ ERROR: casting `f64` to `i32` may truncate the value
        1f32 as u8;
        //~^ ERROR: casting `f32` to `u8` may truncate the value
        //~| ERROR: casting `f32` to `u8` may lose the sign of the value
    }
    // Test clippy::cast_possible_wrap
    1u8 as i8;
    //~^ ERROR: casting `u8` to `i8` may wrap around the value
    //~| NOTE: `-D clippy::cast-possible-wrap` implied by `-D warnings`
    1u16 as i16;
    //~^ ERROR: casting `u16` to `i16` may wrap around the value
    1u32 as i32;
    //~^ ERROR: casting `u32` to `i32` may wrap around the value
    1u64 as i64;
    //~^ ERROR: casting `u64` to `i64` may wrap around the value
    1usize as isize;
    //~^ ERROR: casting `usize` to `isize` may wrap around the value
    // should not wrap, usize is never 8 bits
    1usize as i8;
    //~^ ERROR: casting `usize` to `i8` may truncate the value
    // wraps on 16 bit ptr size
    1usize as i16;
    //~^ ERROR: casting `usize` to `i16` may truncate the value
    //~| ERROR: casting `usize` to `i16` may wrap around the value on targets with 16-bit
    //~| NOTE: `usize` and `isize` may be as small as 16 bits on some platforms
    // wraps on 32 bit ptr size
    1usize as i32;
    //~^ ERROR: casting `usize` to `i32` may truncate the value on targets with 64-bit wid
    //~| ERROR: casting `usize` to `i32` may wrap around the value on targets with 32-bit
    // wraps on 64 bit ptr size
    1usize as i64;
    //~^ ERROR: casting `usize` to `i64` may wrap around the value on targets with 64-bit
    // should not wrap, isize is never 8 bits
    1u8 as isize;
    // wraps on 16 bit ptr size
    1u16 as isize;
    //~^ ERROR: casting `u16` to `isize` may wrap around the value on targets with 16-bit
    //~| NOTE: `usize` and `isize` may be as small as 16 bits on some platforms
    // wraps on 32 bit ptr size
    1u32 as isize;
    //~^ ERROR: casting `u32` to `isize` may wrap around the value on targets with 32-bit
    // wraps on 64 bit ptr size
    1u64 as isize;
    //~^ ERROR: casting `u64` to `isize` may truncate the value on targets with 32-bit wid
    //~| ERROR: casting `u64` to `isize` may wrap around the value on targets with 64-bit
    // Test clippy::cast_sign_loss
    1i32 as u32;
    -1i32 as u32;
    //~^ ERROR: casting `i32` to `u32` may lose the sign of the value
    1isize as usize;
    -1isize as usize;
    //~^ ERROR: casting `isize` to `usize` may lose the sign of the value
    0i8 as u8;
    i8::MAX as u8;
    i16::MAX as u16;
    i32::MAX as u32;
    i64::MAX as u64;
    i128::MAX as u128;

    (-1i8).saturating_abs() as u8;
    // abs() can return a negative value in release builds
    (i8::MIN).abs() as u8;
    //~^ ERROR: casting `i8` to `u8` may lose the sign of the value
    (-1i16).saturating_abs() as u16;
    (-1i32).saturating_abs() as u32;
    (-1i64).abs() as u64;
    (-1isize).abs() as usize;

    (-1i8).checked_abs().unwrap() as u8;
    (i8::MIN).checked_abs().unwrap() as u8;
    (-1i16).checked_abs().unwrap() as u16;
    (-1i32).checked_abs().unwrap() as u32;
    // SAFETY: -1 is a small number which will always return Some
    (unsafe { (-1i64).checked_abs().unwrap_unchecked() }) as u64;
    (-1isize).checked_abs().expect("-1 is a small number") as usize;

    (-1i8).isqrt() as u8;
    (i8::MIN).isqrt() as u8;
    (-1i16).isqrt() as u16;
    (-1i32).isqrt() as u32;
    (-1i64).isqrt() as u64;
    (-1isize).isqrt() as usize;

    (-1i8).checked_isqrt().unwrap() as u8;
    (i8::MIN).checked_isqrt().unwrap() as u8;
    (-1i16).checked_isqrt().unwrap() as u16;
    (-1i32).checked_isqrt().unwrap() as u32;
    // SAFETY: -1 is a small number which will always return Some
    (unsafe { (-1i64).checked_isqrt().unwrap_unchecked() }) as u64;
    (-1isize).checked_isqrt().expect("-1 is a small number") as usize;

    (-1i8).rem_euclid(1i8) as u8;
    (-1i8).wrapping_rem_euclid(1i8) as u16;
    (-1i16).rem_euclid(1i16) as u16;
    (-1i16).rem_euclid(1i16) as u32;
    (-1i32).rem_euclid(1i32) as u32;
    (-1i32).rem_euclid(1i32) as u64;
    (-1i64).rem_euclid(1i64) as u64;
    (-1i64).rem_euclid(1i64) as u128;
    (-1isize).rem_euclid(1isize) as usize;
    (1i8).rem_euclid(-1i8) as u8;
    (1i8).wrapping_rem_euclid(-1i8) as u16;
    (1i16).rem_euclid(-1i16) as u16;
    (1i16).rem_euclid(-1i16) as u32;
    (1i32).rem_euclid(-1i32) as u32;
    (1i32).rem_euclid(-1i32) as u64;
    (1i64).rem_euclid(-1i64) as u64;
    (1i64).rem_euclid(-1i64) as u128;
    (1isize).rem_euclid(-1isize) as usize;

    (-1i8).checked_rem_euclid(1i8).unwrap() as u8;
    (-1i8).checked_rem_euclid(1i8).unwrap() as u16;
    (-1i16).checked_rem_euclid(1i16).unwrap() as u16;
    (-1i16).checked_rem_euclid(1i16).unwrap() as u32;
    (-1i32).checked_rem_euclid(1i32).unwrap() as u32;
    (-1i32).checked_rem_euclid(1i32).unwrap() as u64;
    (-1i64).checked_rem_euclid(1i64).unwrap() as u64;
    (-1i64).checked_rem_euclid(1i64).unwrap() as u128;
    (-1isize).checked_rem_euclid(1isize).unwrap() as usize;
    (1i8).checked_rem_euclid(-1i8).unwrap() as u8;
    (1i8).checked_rem_euclid(-1i8).unwrap() as u16;
    (1i16).checked_rem_euclid(-1i16).unwrap() as u16;
    (1i16).checked_rem_euclid(-1i16).unwrap() as u32;
    (1i32).checked_rem_euclid(-1i32).unwrap() as u32;
    (1i32).checked_rem_euclid(-1i32).unwrap() as u64;
    (1i64).checked_rem_euclid(-1i64).unwrap() as u64;
    (1i64).checked_rem_euclid(-1i64).unwrap() as u128;
    (1isize).checked_rem_euclid(-1isize).unwrap() as usize;

    // no lint for `cast_possible_truncation`
    // with `signum` method call (see issue #5395)
    let x: i64 = 5;
    let _ = x.signum() as i32;

    let s = x.signum();
    let _ = s as i32;

    // Test for signed min
    // should be linted because signed
    (-99999999999i64).min(1) as i8;
    //~^ ERROR: casting `i64` to `i8` may truncate the value

    // Test for various operations that remove enough bits for the result to fit
    (999999u64 & 1) as u8;
    (999999u64 % 15) as u8;
    (999999u64 / 0x1_0000_0000_0000) as u16;
    ({ 999999u64 >> 56 }) as u8;
    ({
        let x = 999999u64;
        x.min(1)
    }) as u8;
    999999u64.clamp(0, 255) as u8;
    // should still be linted
    999999u64.clamp(0, 256) as u8;
    //~^ ERROR: casting `u64` to `u8` may truncate the value

    #[derive(Clone, Copy)]
    enum E1 {
        A,
        B,
        C,
    }
    impl E1 {
        fn test(self) {
            // Don't lint. `0..=2` fits in u8
            let _ = self as u8;
        }
    }

    #[derive(Clone, Copy)]
    enum E2 {
        A = 255,
        B,
    }
    impl E2 {
        fn test(self) {
            let _ = self as u8;
            //~^ ERROR: casting `main::E2` to `u8` may truncate the value
            let _ = Self::B as u8;
            //~^ ERROR: casting `main::E2::B` to `u8` will truncate the value
            //~| NOTE: `-D clippy::cast-enum-truncation` implied by `-D warnings`
            // Don't lint. `255..=256` fits in i16
            let _ = self as i16;
            // Don't lint.
            let _ = Self::A as u8;
        }
    }

    #[derive(Clone, Copy)]
    enum E3 {
        A = -1,
        B,
        C = 50,
    }
    impl E3 {
        fn test(self) {
            // Don't lint. `-1..=50` fits in i8
            let _ = self as i8;
        }
    }

    #[derive(Clone, Copy)]
    enum E4 {
        A = -128,
        B,
    }
    impl E4 {
        fn test(self) {
            // Don't lint. `-128..=-127` fits in i8
            let _ = self as i8;
        }
    }

    #[derive(Clone, Copy)]
    enum E5 {
        A = -129,
        B = 127,
    }
    impl E5 {
        fn test(self) {
            let _ = self as i8;
            //~^ ERROR: casting `main::E5` to `i8` may truncate the value
            let _ = Self::A as i8;
            //~^ ERROR: casting `main::E5::A` to `i8` will truncate the value
            // Don't lint. `-129..=127` fits in i16
            let _ = self as i16;
            // Don't lint.
            let _ = Self::B as u8;
        }
    }

    #[derive(Clone, Copy)]
    #[repr(u32)]
    enum E6 {
        A = u16::MAX as u32,
        B,
    }
    impl E6 {
        fn test(self) {
            let _ = self as i16;
            //~^ ERROR: casting `main::E6` to `i16` may truncate the value
            // Don't lint. `2^16-1` fits in u16
            let _ = Self::A as u16;
            // Don't lint. `2^16-1..=2^16` fits in u32
            let _ = self as u32;
            // Don't lint.
            let _ = Self::A as u16;
        }
    }

    #[derive(Clone, Copy)]
    #[repr(u64)]
    enum E7 {
        A = u32::MAX as u64,
        B,
    }
    impl E7 {
        fn test(self) {
            let _ = self as usize;
            //~^ ERROR: casting `main::E7` to `usize` may truncate the value on targets wi
            // Don't lint.
            let _ = Self::A as usize;
            // Don't lint. `2^32-1..=2^32` fits in u64
            let _ = self as u64;
        }
    }

    #[derive(Clone, Copy)]
    #[repr(i128)]
    enum E8 {
        A = i128::MIN,
        B,
        C = 0,
        D = i128::MAX,
    }
    impl E8 {
        fn test(self) {
            // Don't lint. `-(2^127)..=2^127-1` fits it i128
            let _ = self as i128;
        }
    }

    #[derive(Clone, Copy)]
    #[repr(u128)]
    enum E9 {
        A,
        B = u128::MAX,
    }
    impl E9 {
        fn test(self) {
            // Don't lint.
            let _ = Self::A as u8;
            // Don't lint. `0..=2^128-1` fits in u128
            let _ = self as u128;
        }
    }

    #[derive(Clone, Copy)]
    #[repr(usize)]
    enum E10 {
        A,
        B = u32::MAX as usize,
    }
    impl E10 {
        fn test(self) {
            let _ = self as u16;
            //~^ ERROR: casting `main::E10` to `u16` may truncate the value
            // Don't lint.
            let _ = Self::B as u32;
            // Don't lint.
            let _ = self as u64;
        }
    }
}

fn avoid_subtract_overflow(q: u32) {
    let c = (q >> 16) as u8;
    //~^ ERROR: casting `u32` to `u8` may truncate the value
    c as usize;

    let c = (q / 1000) as u8;
    //~^ ERROR: casting `u32` to `u8` may truncate the value
    c as usize;
}

fn issue11426() {
    (&42u8 >> 0xa9008fb6c9d81e42_0e25730562a601c8_u128) as usize;
}

fn issue11642() {
    fn square(x: i16) -> u32 {
        let x = x as i32;
        (x * x) as u32;
        x.pow(2) as u32;
        (-2_i32).saturating_pow(2) as u32
    }

    let _a = |x: i32| -> u32 { (x * x * x * x) as u32 };

    (2_i32).checked_pow(3).unwrap() as u32;
    (-2_i32).pow(3) as u32;
    //~^ ERROR: casting `i32` to `u32` may lose the sign of the value

    (3_i32 % 2) as u32;
    (3_i32 % -2) as u32;
    (-5_i32 % 2) as u32;
    //~^ ERROR: casting `i32` to `u32` may lose the sign of the value
    (-5_i32 % -2) as u32;
    //~^ ERROR: casting `i32` to `u32` may lose the sign of the value
    (2_i32 >> 1) as u32;
    (-2_i32 >> 1) as u32;
    //~^ ERROR: casting `i32` to `u32` may lose the sign of the value

    let x: i32 = 10;
    (x * x) as u32;
    (x * x * x) as u32;
    //~^ ERROR: casting `i32` to `u32` may lose the sign of the value

    let y: i16 = -2;
    (y * y * y * y * -2) as u16;
    //~^ ERROR: casting `i16` to `u16` may lose the sign of the value
    (y * y * y / y * 2) as u16;
    (y * y / y * 2) as u16;
    //~^ ERROR: casting `i16` to `u16` may lose the sign of the value
    (y / y * y * -2) as u16;
    //~^ ERROR: casting `i16` to `u16` may lose the sign of the value

    (y + y + y + -2) as u16;
    //~^ ERROR: casting `i16` to `u16` may lose the sign of the value
    (y + y + y + 2) as u16;
    //~^ ERROR: casting `i16` to `u16` may lose the sign of the value

    let z: i16 = 2;
    (z + -2) as u16;
    //~^ ERROR: casting `i16` to `u16` may lose the sign of the value
    (z + z + 2) as u16;

    fn foo(a: i32, b: i32, c: i32) -> u32 {
        (a * a * b * b * c * c) as u32;
        (a * b * c) as u32;
        //~^ ERROR: casting `i32` to `u32` may lose the sign of the value
        (a * -b * c) as u32;
        //~^ ERROR: casting `i32` to `u32` may lose the sign of the value
        (a * b * c * c) as u32;
        (a * -2) as u32;
        //~^ ERROR: casting `i32` to `u32` may lose the sign of the value
        (a * b * c * -2) as u32;
        //~^ ERROR: casting `i32` to `u32` may lose the sign of the value
        (a / b) as u32;
        (a / b * c) as u32;
        //~^ ERROR: casting `i32` to `u32` may lose the sign of the value
        (a / b + b * c) as u32;
        //~^ ERROR: casting `i32` to `u32` may lose the sign of the value
        a.saturating_pow(3) as u32;
        //~^ ERROR: casting `i32` to `u32` may lose the sign of the value
        (a.abs() * b.pow(2) / c.abs()) as u32
        //~^ ERROR: casting `i32` to `u32` may lose the sign of the value
    }
}
