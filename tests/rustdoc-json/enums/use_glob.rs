// Regression test for <https://github.com/rust-lang/rust/issues/104942>

//@ jq_set color = '.index[] | select(.name == "Color").id'
pub enum Color {
    Red,
    Green,
    Blue,
}

//@ jq_set use_color = '.index[] | select(.inner.use).id'
//@ jq_is '.index[] | select(.inner.use).inner.use.id' $color
//@ jq_is '.index[] | select(.inner.use).inner.use.is_glob' true
pub use Color::*;

//@ jq_ismany '.index[] | select(.name == "use_glob").inner.module.items[]' $color $use_color
