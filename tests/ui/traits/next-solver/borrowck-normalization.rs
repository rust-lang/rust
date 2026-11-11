//@ check-pass
//@ compile-flags: -Znext-solver
//! https://github.com/rust-lang/rust/pull/160443 reworked how normalization works in type
//! relations. Initially, that PR left out normalization in NllTypeRelating, as it seemed to be
//! unused. However, upon doing a stage 2 build with -Znext-solver, turns out it *is* used. This is
//! the extracted case from the failing crate, to have it as a proper test instead of just failing
//! to compile stage 2.

trait Trait {
    type Item;
}

struct Struct<I: Trait> {
    item: I::Item,
}

impl<I: Trait> Struct<I> {
    fn func(self) {
        let Self { item } = self;
    }
}

fn main() {}
