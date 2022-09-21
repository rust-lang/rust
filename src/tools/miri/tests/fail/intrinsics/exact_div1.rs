#![feature(core_intrinsics)]
fn main() {
    // divison by 0
    unsafe { std::intrinsics::exact_div(2, 0) }; //~ ERROR: divisor of zero
}
