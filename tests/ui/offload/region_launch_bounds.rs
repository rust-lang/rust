//@ run-fail
//@ edition: 2024
//@ aux-crate: offload_strategies=offload_strategies.rs
//@ error-pattern: offload launch is not supported by the region's partitioning strategy

#![feature(gpu_offload)]

use core::offload::{Region, RegionLaunchCheck};
use offload_strategies::Linear1D;

fn main() {
    let mut x = [0.0f32; 4];
    let region = Region::<f32, Linear1D>::new(&mut x[..]);

    // `Linear1D` only supports single-dimensional launches, so `[4, 2, 1]` is
    // rejected.
    region.__offload_check_launch([4, 2, 1], [1, 1, 1]);
}
