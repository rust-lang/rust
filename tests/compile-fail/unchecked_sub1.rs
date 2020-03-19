#![feature(core_intrinsics)]
fn main() {
    // MIN overflow
    unsafe { std::intrinsics::unchecked_sub(14u32, 22); } //~ ERROR overflow executing `unchecked_sub`
}
