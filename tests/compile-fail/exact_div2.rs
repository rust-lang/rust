#![feature(core_intrinsics)]
fn main() {
    // divison with a remainder
    unsafe { std::intrinsics::exact_div(2u16, 3); } //~ ERROR 2 cannot be divided by 3 without remainder
}
