#![feature(no_core)]
#![no_core]

// @set AlwaysNone = "$.index[*][?(@.name == 'AlwaysNone')].id"
pub enum AlwaysNone {
    // @set None = "$.index[*][?(@.name == 'None')].id"
    None,
}
// @is "$.index[*][?(@.name == 'AlwaysNone')].inner.enum.variants[*]" $None

// @set use_None = "$.index[*][?(@.inner.import)].id"
// @is "$.index[*][?(@.inner.import)].inner.import.id" $None
pub use AlwaysNone::None;

// @ismany "$.index[*][?(@.name == 'use_variant')].inner.module.items[*]" $AlwaysNone $use_None
