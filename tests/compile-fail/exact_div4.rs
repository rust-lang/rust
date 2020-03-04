#![feature(core_intrinsics)]
fn main() {
    // divison of MIN by -1
    unsafe { std::intrinsics::exact_div(i64::MIN, -1); } //~ ERROR result of dividing MIN by -1 cannot be represented
}
