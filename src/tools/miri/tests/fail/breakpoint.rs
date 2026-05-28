#![feature(core_intrinsics)]

fn main() {
    core::intrinsics::breakpoint(); //~ ERROR: trace/breakpoint trap
}
