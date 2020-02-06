#![feature(core_intrinsics)]
fn main() {
    // MAX overflow
    unsafe { std::intrinsics::unchecked_add(40000u16, 30000); } //~ ERROR Overflow executing `unchecked_add`
}
