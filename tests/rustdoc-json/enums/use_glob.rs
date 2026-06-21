// Regression test for <https://github.com/rust-lang/rust/issues/104942>

//@ jqset color = '.index[] | select(.name == "Color").id'
pub enum Color {
    Red,
    Green,
    Blue,
}

//@ jqset use_color = '.index[] | select(.inner.use)'
//@ jq '$use_color.inner.use.id? == $color'
//@ jq '$use_color.inner.use.is_glob? == true'
pub use Color::*;

//@ jq '.index["\(.root)"].inner.module.items? | inside([$color, $use_color.id])'
