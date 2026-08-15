#![no_std]
pub fn drop_default<T: core::default::Default>(_x: T) {}

//@ !has "$.index[?(@.name=='Debug')]"
//@ !has "$.index[?(@.name=='Default')]"
