#![feature(optimize_attribute)]

//@ jq .index[] | select(.name == "speed").attrs == ["#[attr = Optimize(Speed)]"]
#[optimize(speed)]
pub fn speed() {}

//@ jq .index[] | select(.name == "size").attrs == ["#[attr = Optimize(Size)]"]
#[optimize(size)]
pub fn size() {}

//@ jq .index[] | select(.name == "none").attrs == ["#[attr = Optimize(DoNotOptimize)]"]
#[optimize(none)]
pub fn none() {}
