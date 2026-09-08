#![feature(abi_gpu_kernel, rustc_attrs, no_core)]
#![no_core]
#![crate_type = "rlib"]

extern crate minicore;

// Partitioning assigns items to codegen units by module, so with `-Ccodegen-units=2` these two
// kernels would land in separate CGUs.
pub mod first {
    #[unsafe(no_mangle)]
    #[rustc_offload_kernel]
    pub unsafe extern "gpu-kernel" fn kernel_in_first_module(x: *mut f32, k: f32) {
        unsafe { *x = k };
    }
}

pub mod second {
    #[unsafe(no_mangle)]
    #[rustc_offload_kernel]
    pub unsafe extern "gpu-kernel" fn kernel_in_second_module(x: *mut f32, k: f32) {
        unsafe { *x = k };
    }
}
