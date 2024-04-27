// Test that the crate hash is not affected by reordering items.

//@ revisions:rpass1 rpass2 rpass3
//@ compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]

// Check that reordering otherwise identical items is not considered a
// change at all.
#[rustc_clean(cfg = "rpass2")]
// But removing an item, naturally, is.
#[rustc_clean(except="hir_crate", cfg = "rpass3")]
#[cfg(rpass1)]
pub struct X {
    pub x: u32,
}

pub struct Y {
    pub x: u32,
}

#[cfg(rpass2)]
pub struct X {
    pub x: u32,
}

pub fn main() {}
