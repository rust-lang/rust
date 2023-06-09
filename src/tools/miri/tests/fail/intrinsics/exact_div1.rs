#![feature(core_intrinsics)]
fn main() {
    // division by 0
    unsafe { std::intrinsics::exact_div(2, 0) }; //~ ERROR: divisor of zero
}
