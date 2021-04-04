#![feature(rustc_attrs, extended_key_value_attributes)]

#[rustc_dummy = 1usize] //~ ERROR: suffixed literals are not allowed in attributes
#[rustc_dummy = 1u8] //~ ERROR: suffixed literals are not allowed in attributes
#[rustc_dummy = 1u16] //~ ERROR: suffixed literals are not allowed in attributes
#[rustc_dummy = 1u32] //~ ERROR: suffixed literals are not allowed in attributes
#[rustc_dummy = 1u64] //~ ERROR: suffixed literals are not allowed in attributes
#[rustc_dummy = 1isize] //~ ERROR: suffixed literals are not allowed in attributes
#[rustc_dummy = 1i8] //~ ERROR: suffixed literals are not allowed in attributes
#[rustc_dummy = 1i16] //~ ERROR: suffixed literals are not allowed in attributes
#[rustc_dummy = 1i32] //~ ERROR: suffixed literals are not allowed in attributes
#[rustc_dummy = 1i64] //~ ERROR: suffixed literals are not allowed in attributes
#[rustc_dummy = 1.0f32] //~ ERROR: suffixed literals are not allowed in attributes
#[rustc_dummy = 1.0f64] //~ ERROR: suffixed literals are not allowed in attributes
fn main() {}
