pub const A: i8 = -i8::MIN; //~ ERROR constant
pub const B: u8 = 200u8 + 200u8; //~ ERROR constant
pub const C: u8 = 200u8 * 4; //~ ERROR constant
pub const D: u8 = 42u8 - (42u8 + 1); //~ ERROR constant
pub const E: u8 = [5u8][1]; //~ ERROR constant

fn main() {
    let _a = A;
    let _b = B;
    let _c = C;
    let _d = D;
    let _e = E;
    let _e = [6u8][1];
}
