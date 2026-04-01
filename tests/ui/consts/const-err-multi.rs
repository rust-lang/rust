pub const A: i8 = -i8::MIN;
//~^ NOTE failed here
//~| ERROR attempt to negate `i8::MIN`, which would overflow
pub const B: i8 = A;
//~^ NOTE erroneous constant
pub const C: u8 = A as u8;
//~^ NOTE erroneous constant
pub const D: i8 = 50 - A;
//~^ NOTE erroneous constant

fn main() {
    let _ = (A, B, C, D);
}
