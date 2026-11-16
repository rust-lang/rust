#![feature(gpu_offload, rustc_attrs)]
#![allow(internal_features)]
#![crate_type = "rlib"]

#[rustc_offload_kernel]
pub fn generic_kernel<T: Copy>(x: T) {}

#[rustc_offload_kernel]
pub fn kernel(x: i32) {}
