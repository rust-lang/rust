#![deny(const_err)]

pub const A: i8 = -i8::MIN;
//~^ ERROR const_err
pub const B: i8 = A;
//~^ ERROR const_err
pub const C: u8 = A as u8;
//~^ ERROR const_err
pub const D: i8 = 50 - A;
//~^ ERROR const_err

fn main() {
    let _ = (A, B, C, D);
}
