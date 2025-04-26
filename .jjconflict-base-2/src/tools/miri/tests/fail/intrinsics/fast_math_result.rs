#![feature(core_intrinsics)]

fn main() {
    unsafe {
        let _x: f32 = core::intrinsics::fdiv_fast(1.0, 0.0); //~ ERROR: `fdiv_fast` intrinsic produced non-finite value as result
    }
}
