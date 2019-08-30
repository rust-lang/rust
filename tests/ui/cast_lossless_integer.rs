// run-rustfix

#![allow(clippy::no_effect, clippy::unnecessary_operation, dead_code)]
#![warn(clippy::cast_lossless)]

fn main() {
    // Test clippy::cast_lossless with casts to integer types
    1i8 as i16;
    1i8 as i32;
    1i8 as i64;
    1u8 as i16;
    1u8 as i32;
    1u8 as i64;
    1u8 as u16;
    1u8 as u32;
    1u8 as u64;
    1i16 as i32;
    1i16 as i64;
    1u16 as i32;
    1u16 as i64;
    1u16 as u32;
    1u16 as u64;
    1i32 as i64;
    1u32 as i64;
    1u32 as u64;

    // Test with an expression wrapped in parens
    (1u8 + 1u8) as u16;
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
