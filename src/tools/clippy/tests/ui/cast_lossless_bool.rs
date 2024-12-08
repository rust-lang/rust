#![allow(dead_code)]
#![warn(clippy::cast_lossless)]

type U8 = u8;

fn main() {
    // Test clippy::cast_lossless with casts to integer types
    let _ = true as u8;
    let _ = true as u16;
    let _ = true as u32;
    let _ = true as u64;
    let _ = true as u128;
    let _ = true as usize;

    let _ = true as i8;
    let _ = true as i16;
    let _ = true as i32;
    let _ = true as i64;
    let _ = true as i128;
    let _ = true as isize;

    // Test with an expression wrapped in parens
    let _ = (true | false) as u16;

    let _ = true as U8;
}

// The lint would suggest using `u32::from(input)` here but the `XX::from` function is not const,
// so we skip the lint if the expression is in a const fn.
// See #3656
const fn abc(input: bool) -> u32 {
    input as u32
}

// Same as the above issue. We can't suggest `::from` in const fns in impls
mod cast_lossless_in_impl {
    struct A;

    impl A {
        pub const fn convert(x: bool) -> u64 {
            x as u64
        }
    }
}

#[clippy::msrv = "1.27"]
fn msrv_1_27() {
    let _ = true as u8;
}

#[clippy::msrv = "1.28"]
fn msrv_1_28() {
    let _ = true as u8;
}
