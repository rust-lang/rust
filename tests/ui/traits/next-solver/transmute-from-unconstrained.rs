//! Regression test for <https://github.com/rust-lang/rust/issues/151314>.
//!
//! Calling a function with an unconstrained `TransmuteFrom` obligation used to
//! trigger a `layout_of: unexpected type` ICE under the next-gen trait solver
//! instead of reporting that type annotations are needed.

//@ compile-flags: -Znext-solver=globally

#![feature(transmutability)]

fn assert_transmutable<T>()
where
    (): std::mem::TransmuteFrom<T>,
{
}

fn main() {
    assert_transmutable()
    //~^ ERROR type annotations needed
}
