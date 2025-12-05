#![feature(core_intrinsics, funnel_shifts)]

fn main() {
    unsafe {
        std::intrinsics::unchecked_funnel_shr(1_u32, 2, 32); //~ ERROR: Undefined Behavior
    }
}
