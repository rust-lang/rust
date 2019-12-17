#![deny(const_err)]

pub const A: i8 = -i8::MIN; //~ ERROR const_err
pub const B: u8 = 200u8 + 200u8; //~ ERROR const_err
pub const C: u8 = 200u8 * 4; //~ ERROR const_err
pub const D: u8 = 42u8 - (42u8 + 1); //~ ERROR const_err
pub const E: u8 = [5u8][1]; //~ ERROR const_err

fn main() {
    let _a = A;
    let _b = B;
    let _c = C;
    let _d = D;
    let _e = E;
    let _e = [6u8][1];
}
