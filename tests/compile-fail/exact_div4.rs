#![feature(core_intrinsics)]
fn main() {
    // divison of min_value by -1
    unsafe { std::intrinsics::exact_div(i64::min_value(), -1); } //~ ERROR result of dividing MIN by -1 cannot be represented
}
