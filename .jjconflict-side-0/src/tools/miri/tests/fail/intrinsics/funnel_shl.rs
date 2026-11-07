#![feature(core_intrinsics)]

fn main() {
    unsafe {
        std::intrinsics::unchecked_funnel_shl(1_u32, 2, 32); //~ ERROR: Undefined Behavior
    }
}
