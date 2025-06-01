//@ compile-flags: --crate-type=lib
//@ edition: 2018
#![feature(abi_gpu_kernel)]
struct Test;

impl Test {
    pub extern "gpu-kernel" fn test(val: &str) {}
    //~^ ERROR [E0570]
}
