#![feature(core_intrinsics)]
fn main() {
    // MAX overflow
    unsafe { std::intrinsics::unchecked_mul(300u16, 250u16); } //~ ERROR Overflow executing `unchecked_mul`
}
