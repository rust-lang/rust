//@ compile-flags: -Zunstable-options -Zoffload=Device -Clto=fat
//@ edition: 2024
//@ aux-crate: offload_strategies=offload_strategies.rs

// This tests ensures an error is emmited with passing a `&Region<'_, _, _>` args to offload.

#![feature(core_intrinsics)]
#![feature(gpu_offload)]
#![feature(offload)]

use core::offload::Region;
use offload_strategies::Dummy;

fn kernel_shared(_region: &Region<'_, f32, Dummy>) {}

fn kernel_mut(_region: &mut Region<'_, f32, Dummy>) {}

fn main() {
    let mut x = [0.0f32; 4];
    let region = Region::<f32, Dummy>::new(&mut x[..]);
    core::intrinsics::offload::<_, _, ()>(
        //~^ ERROR offload kernel argument 0 is a reference to a `Region`
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
        //~^ ERROR offload kernel argument 0 is a reference to a `Region`
        kernel_mut,
        [1, 1, 1],
        [1, 1, 1],
        0,
        -1,
        (&mut region,),
    );
}
