#![feature(no_core)]
#![no_core]

// @set AlwaysNone = "$.index[*][?(@.name == 'AlwaysNone')].id"
pub enum AlwaysNone {
    // @set None = "$.index[*][?(@.name == 'None')].id"
    None,
}
// @is "$.index[*][?(@.name == 'AlwaysNone')].inner.variants[*]" $None

// @set use_None = "$.index[*][?(@.kind == 'import')].id"
// @is "$.index[*][?(@.kind == 'import')].inner.id" $None
pub use AlwaysNone::None;

// @ismany "$.index[*][?(@.name == 'use_variant')].inner.items[*]" $AlwaysNone $use_None
