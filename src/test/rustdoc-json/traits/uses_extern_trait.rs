#![no_std]
pub fn drop_default<T: core::default::Default>(_x: T) {}

// FIXME(adotinthevoid): Theses shouldn't be here
// @has "$.index[*][?(@.name=='Debug')]"
// @set Debug_fmt = "$.index[*][?(@.name=='Debug')].inner.items[*]"
// @has "$.index[*][?(@.name=='fmt')].id" $Debug_fmt
