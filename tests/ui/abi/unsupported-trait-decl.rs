//@ compile-flags: --crate-type=lib

#![feature(abi_gpu_kernel)]

trait T {
    extern "gpu-kernel" fn mu();
    //~^ ERROR[E0570]
}

type TAU = extern "gpu-kernel" fn();
//~^ ERROR[E0570]
