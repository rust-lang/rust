#![feature(core_intrinsics)]
fn main() {
    // MIN overflow
    unsafe { std::intrinsics::unchecked_mul(1_000_000_000i32, -4); } //~ ERROR Overflow executing `unchecked_mul`
}
