#![feature(gpu_offload, rustc_attrs)]
#![allow(internal_features)]
#![crate_type = "rlib"]

#[cfg(not(device))]
#[inline(never)]
pub fn launch_f32(x: f32) {
    core::offload::offload! {
        kernel = generic_kernel::<f32>,
        args = (x,),
    }
}

#[cfg(not(device))]
#[inline(never)]
pub fn launch_f64(x: f64) {
    core::offload::offload! {
        kernel = generic_kernel::<f64>,
        args = (x,),
    }
}

#[cfg(not(device))]
#[inline(never)]
pub fn launch(x: i32) {
    core::offload::offload! {
        kernel = kernel,
        args = (x,),
    }
}

#[rustc_offload_kernel]
fn generic_kernel<T: Copy>(_x: T) {}

#[rustc_offload_kernel]
fn kernel(_x: i32) {}
