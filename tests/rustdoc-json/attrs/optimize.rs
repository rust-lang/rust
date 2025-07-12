#![feature(optimize_attribute)]

//@ eq .index[] | select(.name == "speed").attrs | ., ["#[attr = Optimize(Speed)]"]
#[optimize(speed)]
pub fn speed() {}

//@ eq .index[] | select(.name == "size").attrs | ., ["#[attr = Optimize(Size)]"]
#[optimize(size)]
pub fn size() {}

//@ eq .index[] | select(.name == "none").attrs | ., ["#[attr = Optimize(DoNotOptimize)]"]
#[optimize(none)]
pub fn none() {}
