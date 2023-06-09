// Regression test for <https://github.com/rust-lang/rust/issues/104942>

#![feature(no_core)]
#![no_core]

// @set Color = "$.index[*][?(@.name == 'Color')].id"
pub enum Color {
    Red,
    Green,
    Blue,
}

// @set use_Color = "$.index[*][?(@.inner.import)].id"
// @is "$.index[*][?(@.inner.import)].inner.import.id" $Color
// @is "$.index[*][?(@.inner.import)].inner.import.glob" true
pub use Color::*;

// @ismany "$.index[*][?(@.name == 'use_glob')].inner.module.items[*]" $Color $use_Color
