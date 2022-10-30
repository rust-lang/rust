#![feature(core_intrinsics)]

fn main() {
    unsafe {
        core::intrinsics::breakpoint() //~ ERROR: Trace/breakpoint trap
    };
}
