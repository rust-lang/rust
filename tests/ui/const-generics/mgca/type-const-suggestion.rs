//! Regression test for <https://github.com/rust-lang/rust/issues/151602>.

#![feature(min_generic_const_args)]

trait Trait {
    type const K: i32;
}
fn take(_: impl Trait<0>) {}
//~^ ERROR: trait takes 0 generic arguments but 1 generic argument was supplied [E0107]

fn main() {}
