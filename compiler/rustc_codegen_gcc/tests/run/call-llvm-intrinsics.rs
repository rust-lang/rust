// Compiler:
//
// Run-time:
//   status: 0

// FIXME: Remove this test once rustc's `./tests/codegen/riscv-abi/call-llvm-intrinsics.rs`
// stops ignoring GCC backend.

#![feature(link_llvm_intrinsics)]
#![allow(internal_features)]

struct A;

impl Drop for A {
    fn drop(&mut self) {
        println!("A");
    }
}

extern "C" {
    #[link_name = "llvm.sqrt.f32"]
    fn sqrt(x: f32) -> f32;
}

pub fn do_call() {
    let _a = A;

    unsafe {
        // Ensure that we `call` LLVM intrinsics instead of trying to `invoke` them
        // CHECK: store float 4.000000e+00, float* %{{.}}, align 4
        // CHECK: call float @llvm.sqrt.f32(float %{{.}}
        sqrt(4.0);
    }
}

fn main() {
    do_call();
}
