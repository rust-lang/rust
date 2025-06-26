#![no_std]

//@ eq .index[] | select(.name == "Aligned").attrs | ., ["#[repr(align(4))]"]
#[repr(align(4))]
pub struct Aligned {
    a: i8,
    b: i64,
}
