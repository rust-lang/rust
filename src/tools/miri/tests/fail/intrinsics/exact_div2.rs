#![feature(core_intrinsics)]
fn main() {
    // division with a remainder
    unsafe { std::intrinsics::exact_div(2u16, 3) }; //~ ERROR: 2_u16 cannot be divided by 3_u16 without remainder
}
