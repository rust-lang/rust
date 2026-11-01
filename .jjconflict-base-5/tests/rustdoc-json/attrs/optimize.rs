#![feature(optimize_attribute)]

//@ is "$.index[?(@.name=='speed')].attrs" '[{"other": "#[attr = Optimize(Speed)]"}]'
#[optimize(speed)]
pub fn speed() {}

//@ is "$.index[?(@.name=='size')].attrs" '[{"other": "#[attr = Optimize(Size)]"}]'
#[optimize(size)]
pub fn size() {}

//@ is "$.index[?(@.name=='none')].attrs" '[{"other": "#[attr = Optimize(DoNotOptimize)]"}]'
#[optimize(none)]
pub fn none() {}
