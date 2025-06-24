#![feature(optimize_attribute)]

//@ is "$.index[?(@.name=='speed')].attrs" '["#[attr = Optimize(Speed)]"]'
#[optimize(speed)]
pub fn speed() {}

//@ is "$.index[?(@.name=='size')].attrs" '["#[attr = Optimize(Size)]"]'
#[optimize(size)]
pub fn size() {}

//@ is "$.index[?(@.name=='none')].attrs" '["#[attr = Optimize(DoNotOptimize)]"]'
#[optimize(none)]
pub fn none() {}
