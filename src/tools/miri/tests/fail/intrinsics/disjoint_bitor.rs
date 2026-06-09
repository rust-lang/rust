#![feature(core_intrinsics)]
fn main() {
    // one bit in common
    unsafe { std::intrinsics::disjoint_bitor(0b01101001_u8, 0b10001110) }; //~ ERROR: Undefined Behavior
}
