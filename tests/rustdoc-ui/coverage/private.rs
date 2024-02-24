//@ compile-flags:-Z unstable-options --show-coverage --document-private-items
//@ check-pass

#![allow(unused)]

//! when `--document-private-items` is passed, nothing is safe. everything must have docs or your
//! score will suffer the consequences

mod this_mod {
    fn private_fn() {}
}

/// See, our public items have docs!
pub struct SomeStruct {
    /// Look, all perfectly documented!
    pub field: usize,
    other: usize,
}

/// Nothing shady going on here. Just a bunch of well-documented code. (cough)
pub fn public_fn() {}
