#![feature(gpu_offload, rustc_attrs)]
#![allow(internal_features)]
#![cfg_attr(device, no_main)]

#[rustc_offload_kernel]
fn kernel<T: Copy>(x: T) {}

#[cfg(not(device))]
fn main() {
    core::offload::offload! {
        kernel = kernel::<f32>,
        args = (0.0f32,),
    }
    core::offload::offload! {
        kernel = kernel::<i32>,
        args = (0i32,),
    }
}
