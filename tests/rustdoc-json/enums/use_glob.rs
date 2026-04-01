// Regression test for <https://github.com/rust-lang/rust/issues/104942>

//@ set Color = "$.index[?(@.name == 'Color')].id"
pub enum Color {
    Red,
    Green,
    Blue,
}

//@ set use_Color = "$.index[?(@.inner.use)].id"
//@ is "$.index[?(@.inner.use)].inner.use.id" $Color
//@ is "$.index[?(@.inner.use)].inner.use.is_glob" true
pub use Color::*;

//@ ismany "$.index[?(@.name == 'use_glob')].inner.module.items[*]" $Color $use_Color
