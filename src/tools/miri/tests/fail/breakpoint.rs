#![feature(core_intrinsics)]

fn main() {
    unsafe {
        core::intrinsics::breakpoint() //~ ERROR: trace/breakpoint trap
    };
}
