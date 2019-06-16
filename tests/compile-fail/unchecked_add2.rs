#![feature(core_intrinsics)]
fn main() {
    // MIN overflow
    unsafe { std::intrinsics::unchecked_add(-30000i16, -8000); } //~ ERROR Overflowing arithmetic in unchecked_add
}
