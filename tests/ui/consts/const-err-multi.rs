pub const A: i8 = -i8::MIN;
//~^ NOTE constant
//~| ERROR attempt to negate `i8::MIN`, which would overflow
pub const B: i8 = A;
//~^ NOTE constant
pub const C: u8 = A as u8;
//~^ NOTE constant
pub const D: i8 = 50 - A;
//~^ NOTE constant

fn main() {
    let _ = (A, B, C, D);
}
