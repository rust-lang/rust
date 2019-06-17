#![feature(core_intrinsics)]
fn main() {
    // signed divison with a remainder
    unsafe { std::intrinsics::exact_div(-19i8, 2); } //~ ERROR Scalar(0xed) cannot be divided by Scalar(0x02) without remainder
}
