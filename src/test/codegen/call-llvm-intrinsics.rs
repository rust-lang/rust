// compile-flags: -C no-prepopulate-passes

// ignore-riscv64

#![feature(link_llvm_intrinsics)]
#![crate_type = "lib"]

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
        // CHECK: call float @llvm.sqrt.f32(float 4.000000e+00
        sqrt(4.0);
    }
}
