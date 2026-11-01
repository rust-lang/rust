//@ compile-flags: -Zunstable-options -Zoffload=Device -Clto=fat

#![feature(gpu_offload)]

fn main() {
    // args_ty is not a tuple
    core::offload::offload! { kernel = kernel_0, args = 42 }
    //~^ ERROR `{integer}` is not a tuple
}

fn kernel_0() {}
