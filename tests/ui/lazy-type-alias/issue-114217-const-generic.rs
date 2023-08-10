// Regression test for <https://github.com/rust-lang/rust/issues/114217>.
// It ensures that enabling the lazy_type_alias` feature doesn't prevent
// const generics to work with type aliases.

// compile-flags: --crate-type lib
// check-pass

#![feature(lazy_type_alias)]
//~^ WARN the feature `lazy_type_alias` is incomplete and may not be safe to use

pub type Word = usize;

pub struct IBig(usize);

pub const fn base_as_ibig<const B: Word>() -> IBig {
    IBig(B)
}
