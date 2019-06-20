#![feature(core_intrinsics)]
fn main() {
    // MAX overflow
    unsafe { std::intrinsics::unchecked_sub(30000i16, -7000); } //~ ERROR Overflowing arithmetic in unchecked_sub
}
