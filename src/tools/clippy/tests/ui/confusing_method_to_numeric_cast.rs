#![feature(float_minimum_maximum)]
#![warn(clippy::confusing_method_to_numeric_cast)]

fn main() {
    let _ = u16::max as usize; //~ confusing_method_to_numeric_cast
    let _ = u16::min as usize; //~ confusing_method_to_numeric_cast
    let _ = u16::max_value as usize; //~ confusing_method_to_numeric_cast
    let _ = u16::min_value as usize; //~ confusing_method_to_numeric_cast

    let _ = f32::maximum as usize; //~ confusing_method_to_numeric_cast
    let _ = f32::max as usize; //~ confusing_method_to_numeric_cast
    let _ = f32::minimum as usize; //~ confusing_method_to_numeric_cast
    let _ = f32::min as usize; //~ confusing_method_to_numeric_cast
}
