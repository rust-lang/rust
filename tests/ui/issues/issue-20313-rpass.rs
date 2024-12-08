//@ run-pass
#![allow(dead_code)]
#![feature(link_llvm_intrinsics)]

extern "C" {
    #[link_name = "llvm.sqrt.f32"]
    fn sqrt(x: f32) -> f32;
}

fn main() {}
