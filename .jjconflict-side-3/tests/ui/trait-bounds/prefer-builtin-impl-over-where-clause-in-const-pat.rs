//! Regression test adjacent to <https://github.com/rust-lang/rust/issues/162331>
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

#![feature(const_trait_impl)]
#![feature(const_clone)]

// At the time of writing, `ZERO` is evaluated in an environment containing `g`'s `(u8,): Clone`
// clause. In the new solver, this wasn't dropped in favor of the built-in `(u8,): Clone` impl when
// resolving an instance for `<(u8,) as Clone>::clone`, which resulted in ambiguity.

const ZERO: (u8,) = (0,).clone();

fn g()
where
    (u8,): Clone,
{
    match (0,) {
        ZERO => {}
        _ => {}
    }
}

fn main() {}
