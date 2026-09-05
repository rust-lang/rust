//@ edition: 2024
//@ aux-crate: offload_strategies=offload_strategies.rs

// This test checks that the borrow checker errors when writing into the data a
// `Region` was created from while the `Region` is still alive.

#![feature(gpu_offload)]
#![feature(offload)]
#![allow(unused_assignments)]

use core::offload::Region;
use offload_strategies::Dummy;

fn main() {
    let mut x = [0.0f32; 4];
    let region = Region::<f32, Dummy>::new(&mut x[..]);

    x[0] = 1.0;
    //~^ ERROR cannot assign to `x[_]` because it is borrowed

    let _view = region.get();
}
