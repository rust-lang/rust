//@ compile-flags: -C no-prepopulate-passes

#![feature(intrinsics)]

extern "rust-intrinsic" {
    fn sqrtf32(x: f32) -> f32;
}
// CHECK: @llvm.sqrt.f32(float) #{{[0-9]*}}

fn main() {
    unsafe {
        sqrtf32(0.0f32);
    }
}
