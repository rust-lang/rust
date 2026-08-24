#![feature(core_intrinsics, rustc_attrs)]
#![allow(internal_features)]
#![cfg_attr(device, no_main)]

#[rustc_offload_kernel]
fn kernel<T: Copy>(x: T) {}

#[cfg(not(device))]
fn main() {
    core::intrinsics::offload::<_, _, ()>(kernel::<f32>, [1, 1, 1], [1, 1, 1], 0, (0.0f32,));
    core::intrinsics::offload::<_, _, ()>(kernel::<i32>, [1, 1, 1], [1, 1, 1], 0, (0i32,));
}
