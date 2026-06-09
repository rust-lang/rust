//! Ensure `TransmuteFrom` with `min_generic_const_args` doesn't ICE
//! during well-formedness checking.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/150457>.

//@ check-pass

#![feature(transmutability)]
#![feature(min_generic_const_args)]

use std::mem::{Assume, TransmuteFrom};

struct W<'a>(&'a ());

fn test<'a>()
where
    W<'a>: TransmuteFrom<
        (),
        {
            Assume {
                alignment: const { true },
                lifetimes: const { true },
                safety: const { true },
                validity: true,
            }
        },
    >,
{
}

fn main() {}
