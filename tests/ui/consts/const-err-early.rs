pub const A: i8 = -i8::MIN; //~ ERROR overflow
pub const B: u8 = 200u8 + 200u8; //~ ERROR overflow
pub const C: u8 = 200u8 * 4; //~ ERROR overflow
pub const D: u8 = 42u8 - (42u8 + 1); //~ ERROR overflow
pub const E: u8 = [5u8][1]; //~ ERROR index out of bounds

fn main() {
    let _a = A;
    let _b = B;
    let _c = C;
    let _d = D;
    let _e = E;
    let _e = [6u8][1];
}
