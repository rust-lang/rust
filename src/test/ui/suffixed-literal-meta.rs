#![feature(custom_attribute)]

#[my_attr = 1usize] //~ ERROR: suffixed literals are not allowed in attributes
#[my_attr = 1u8] //~ ERROR: suffixed literals are not allowed in attributes
#[my_attr = 1u16] //~ ERROR: suffixed literals are not allowed in attributes
#[my_attr = 1u32] //~ ERROR: suffixed literals are not allowed in attributes
#[my_attr = 1u64] //~ ERROR: suffixed literals are not allowed in attributes
#[my_attr = 1isize] //~ ERROR: suffixed literals are not allowed in attributes
#[my_attr = 1i8] //~ ERROR: suffixed literals are not allowed in attributes
#[my_attr = 1i16] //~ ERROR: suffixed literals are not allowed in attributes
#[my_attr = 1i32] //~ ERROR: suffixed literals are not allowed in attributes
#[my_attr = 1i64] //~ ERROR: suffixed literals are not allowed in attributes
#[my_attr = 1.0f32] //~ ERROR: suffixed literals are not allowed in attributes
#[my_attr = 1.0f64] //~ ERROR: suffixed literals are not allowed in attributes
fn main() { }
