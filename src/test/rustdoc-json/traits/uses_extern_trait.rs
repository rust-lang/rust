#![no_std]
pub fn drop_default<T: core::default::Default>(_x: T) {}

// FIXME(adotinthevoid): Theses shouldn't be here
// @has "$.index[*][?(@.name=='Debug')]"

// Debug may have several items. All we depend on here the that `fmt` is first. See
// https://github.com/rust-lang/rust/pull/104525#issuecomment-1331087852 for why we
// can't use [*].

// @set Debug_fmt = "$.index[*][?(@.name=='Debug')].inner.items[0]"
// @has "$.index[*][?(@.name=='fmt')].id" $Debug_fmt
