#![allow(clippy::no_effect, clippy::unnecessary_operation, dead_code)]
#![warn(clippy::cast_lossless)]

type I64Alias = i64;

fn main() {
    // Test clippy::cast_lossless with casts to integer types
    0u8 as u16;
    //~^ cast_lossless

    0u8 as i16;
    //~^ cast_lossless

    0u8 as u32;
    //~^ cast_lossless

    0u8 as i32;
    //~^ cast_lossless

    0u8 as u64;
    //~^ cast_lossless

    0u8 as i64;
    //~^ cast_lossless

    0u8 as u128;
    //~^ cast_lossless

    0u8 as i128;
    //~^ cast_lossless

    0u16 as u32;
    //~^ cast_lossless

    0u16 as i32;
    //~^ cast_lossless

    0u16 as u64;
    //~^ cast_lossless

    0u16 as i64;
    //~^ cast_lossless

    0u16 as u128;
    //~^ cast_lossless

    0u16 as i128;
    //~^ cast_lossless

    0u32 as u64;
    //~^ cast_lossless

    0u32 as i64;
    //~^ cast_lossless

    0u32 as u128;
    //~^ cast_lossless

    0u32 as i128;
    //~^ cast_lossless

    0u64 as u128;
    //~^ cast_lossless

    0u64 as i128;
    //~^ cast_lossless

    0i8 as i16;
    //~^ cast_lossless

    0i8 as i32;
    //~^ cast_lossless

    0i8 as i64;
    //~^ cast_lossless

    0i8 as i128;
    //~^ cast_lossless

    0i16 as i32;
    //~^ cast_lossless

    0i16 as i64;
    //~^ cast_lossless

    0i16 as i128;
    //~^ cast_lossless

    0i32 as i64;
    //~^ cast_lossless

    0i32 as i128;
    //~^ cast_lossless

    0i64 as i128;
    //~^ cast_lossless

    // Test with an expression wrapped in parens
    let _ = (1u8 + 1u8) as u16;
    //~^ cast_lossless

    let _ = 1i8 as I64Alias;
    //~^ cast_lossless

    let _: u16 = 0u8 as _;
    //~^ cast_lossless

    let _: i16 = -1i8 as _;
    //~^ cast_lossless

    let _: u16 = (1u8 + 2) as _;
    //~^ cast_lossless

    let _: u32 = 1i8 as u16 as _;
    //~^ cast_lossless
}

// The lint would suggest using `f64::from(input)` here but the `XX::from` function is not const,
// so we skip the lint if the expression is in a const fn.
// See #3656
const fn abc(input: u16) -> u32 {
    input as u32
}

// Same as the above issue. We can't suggest `::from` in const fns in impls
mod cast_lossless_in_impl {
    struct A;

    impl A {
        pub const fn convert(x: u32) -> u64 {
            x as u64
        }
    }
}

#[derive(PartialEq, Debug)]
#[repr(i64)]
enum Test {
    A = u32::MAX as i64 + 1,
}

fn issue11458() {
    macro_rules! sign_cast {
        ($var: ident, $src: ty, $dest: ty) => {
            <$dest>::from_ne_bytes(($var as $src).to_ne_bytes())
        };
    }
    let x = 10_u128;
    let _ = sign_cast!(x, u8, i8) as i32;
    //~^ cast_lossless

    let _ = (sign_cast!(x, u8, i8) + 1) as i32;
    //~^ cast_lossless
}

fn issue12695() {
    macro_rules! in_macro {
        () => {
            1u8 as u32
            //~^ cast_lossless
        };
    }

    let _ = in_macro!();
}

fn ty_from_macro() {
    macro_rules! ty {
        () => {
            u32
        };
    }

    let _ = 0u8 as ty!();
    //~^ cast_lossless
}

const IN_CONST: u64 = 0u8 as u64;
