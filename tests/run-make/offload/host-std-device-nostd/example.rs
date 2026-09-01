#![feature(gpu_offload, rustc_attrs)]
#![allow(internal_features)]
#![cfg_attr(device, no_std)]
#![cfg_attr(device, no_main)]

#[cfg(device)]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[rustc_offload_kernel]
fn kernel() {}

#[cfg(not(device))]
fn main() {
    core::offload::offload! {
        kernel = kernel,
        args = (),
    }
    println!("Hello from Host with std");
}
