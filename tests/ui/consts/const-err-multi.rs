pub const A: i8 = -i8::MIN;
//~^ ERROR constant
pub const B: i8 = A;
//~^ NOTE_NONVIRAL constant
pub const C: u8 = A as u8;
//~^ NOTE_NONVIRAL constant
pub const D: i8 = 50 - A;
//~^ NOTE_NONVIRAL constant

fn main() {
    let _ = (A, B, C, D);
}
