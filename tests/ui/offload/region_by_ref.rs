//@ compile-flags: -Zunstable-options -Zoffload=Device -Clto=fat
//@ edition: 2024
//@ aux-crate: offload_strategies=offload_strategies.rs

// This test ensures an error is emitted when a `Region` is nested inside another
// type instead of being passed by value.

#![feature(core_intrinsics)]
#![feature(gpu_offload)]
#![feature(offload)]

use core::offload::Region;
use offload_strategies::Dummy;

fn kernel_shared(_region: &Region<'_, f32, Dummy>) {}

fn kernel_mut(_region: &mut Region<'_, f32, Dummy>) {}

fn kernel_nested(_arg: (u32, Region<'_, f32, Dummy>)) {}

fn main() {
    let mut x = [0.0f32; 4];
    let region = Region::<f32, Dummy>::new(&mut x[..]);
    core::intrinsics::offload::<_, _, ()>(
        //~^ ERROR offload kernel argument 0 contains a `Region` nested inside another type
        kernel_shared,
        [1, 1, 1],
        [1, 1, 1],
        0,
        -1,
        (&region,),
    );

    let mut y = [0.0f32; 4];
    let mut region = Region::<f32, Dummy>::new(&mut y[..]);
    core::intrinsics::offload::<_, _, ()>(
        //~^ ERROR offload kernel argument 0 contains a `Region` nested inside another type
        kernel_mut,
        [1, 1, 1],
        [1, 1, 1],
        0,
        -1,
        (&mut region,),
    );

    let mut z = [0.0f32; 4];
    let region = Region::<f32, Dummy>::new(&mut z[..]);
    core::intrinsics::offload::<_, _, ()>(
        //~^ ERROR offload kernel argument 0 contains a `Region` nested inside another type
        kernel_nested,
        [1, 1, 1],
        [1, 1, 1],
        0,
        -1,
        ((0u32, region),),
    );
}
