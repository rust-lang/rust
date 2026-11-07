#![feature(core_intrinsics)]
fn main() {
    // signed division with a remainder
    unsafe { std::intrinsics::exact_div(-19i8, 2) }; //~ ERROR: -19_i8 cannot be divided by 2_i8 without remainder
}
