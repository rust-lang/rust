#![feature(gpu_offload)]

fn dummy() {
    #[core::offload::offload_kernel]
    //~^ ERROR macro attributes on statements are unstable
    let mut x = 5;
    //~^ ERROR offload_kernel must be applied to function

    #[core::offload::offload_kernel]
    x = x + 3;
    //~^^  ERROR attributes on expressions are experimental [E0658]
    //~|   ERROR macro attributes on expressions are unstable
    //~^^^ ERROR offload_kernel must be applied to function

    #[core::offload::offload_kernel]
    //~^ ERROR macro attributes on statements are unstable
    let add_one_v2 = |x: u32| -> u32 { x + 1 };
    //~^ ERROR offload_kernel must be applied to function
}

fn main() {}