// gate-test-cfg_attr_multi

#![cfg_attr(all(), warn(nonstandard_style), allow(unused_attributes))]
//~^ ERROR cfg_attr with zero or more than one attributes is experimental
fn main() {}
