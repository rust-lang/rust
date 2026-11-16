#![feature(gpu_offload, rustc_attrs)]
#![allow(internal_features)]
#![cfg_attr(device, no_main)]

// This is needed for the device pass since crate loading isn't propagated from host
extern crate kernels;

#[cfg(not(device))]
fn main() {
    core::offload::offload! {
        kernel = kernels::generic_kernel::<f32>,
        args = (0.0f32,),
    }
    core::offload::offload! {
        kernel = kernels::generic_kernel::<i32>,
        args = (0i32,),
    }
    core::offload::offload! {
        kernel = kernels::kernel,
        args = (0i32,),
    }
}
