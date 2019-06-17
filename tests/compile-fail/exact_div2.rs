#![feature(core_intrinsics)]
fn main() {
    // divison with a remainder
    unsafe { std::intrinsics::exact_div(2u16, 3); } //~ ERROR Scalar(0x0002) cannot be divided by Scalar(0x0003) without remainder
}
