#![feature(gpu_offload, rustc_attrs)]
#![allow(internal_features)]
#![cfg_attr(device, no_main)]

// This is needed for the device pass since crate loading isn't propagated from host
extern crate launch;

#[cfg(not(device))]
use launch::{launch, launch_f32, launch_f64};

#[cfg(not(device))]
fn main() {
    launch_f32(0.0);
    launch_f64(0.0);
    launch(0);
}
