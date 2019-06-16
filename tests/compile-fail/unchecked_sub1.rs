#![feature(core_intrinsics)]
fn main() {
    // MIN overflow
    unsafe { std::intrinsics::unchecked_sub(14u32, 22); } //~ ERROR Overflowing arithmetic in unchecked_sub
}
