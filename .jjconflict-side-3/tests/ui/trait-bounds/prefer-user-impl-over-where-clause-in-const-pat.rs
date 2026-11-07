//! Regression test for <https://github.com/rust-lang/rust/issues/162331>
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// At the time of writing, when resolving an instance for `<u8 as Trait>::N`, the environment
// contains `f`'s `u8: Trait` clause. The old solver dropped the where clause candidate in favor of
// the `impl Trait for u8` candidate, but the new solver didn't, which resulted in ambiguity.

pub trait Trait {
    const N: usize;
}

impl Trait for u8 {
    const N: usize = 0;
}

pub fn f()
where
    u8: Trait,
{
    match 0 {
        <u8 as Trait>::N => {}
        _ => {}
    }
}

fn main() {}
