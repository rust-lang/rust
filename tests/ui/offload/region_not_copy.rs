//@ edition: 2024
//@ aux-crate: offload_strategies=offload_strategies.rs

// This tests checks `Region` doesn't implement Copy.

#![feature(gpu_offload)]
#![feature(offload)]

use core::offload::Region;
use offload_strategies::Dummy;

fn main() {
    let mut x = [0.0f32; 4];
    let mut a = Region::<f32, Dummy>::new(&mut x[..]);
    let mut b = a;
    if let (Some(_), Some(_)) = (a.get_mut(), b.get_mut()) {
        //~^ ERROR borrow of moved value: `a`
    }
}
