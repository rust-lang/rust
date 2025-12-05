#![feature(core_intrinsics)]

fn main() {
    unsafe {
        let _x: f32 = core::intrinsics::frem_fast(f32::NAN, 3.2); //~ ERROR: `frem_fast` intrinsic called with non-finite value as first parameter
    }
}
