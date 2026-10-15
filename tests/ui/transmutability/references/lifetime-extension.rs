//! Regression test for <https://github.com/rust-lang/rust/issues/164246>.

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ [next] compile-flags: -Znext-solver
//@ [next] check-pass TO BE FIXED

#![feature(transmutability)]

fn func(n: &i32) -> &'static i32 {
    unsafe { core::mem::TransmuteFrom::<_, { core::mem::Assume::NOTHING }>::transmute(n) }
    //[current]~^ ERROR lifetime may not live long enough
}

fn main() {}
