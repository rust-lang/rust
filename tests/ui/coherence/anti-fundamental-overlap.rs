//@ aux-build: anti_fundamental_trait_lib.rs
//@ dont-require-annotations: NOTE

// Test that `#[rustc_anti_fundamental]` prevents reporting
// "downstream crates may implement trait" ambiguity notes,
// and instead reports that only upstream crates can add such an impl.

extern crate anti_fundamental_trait_lib;

use anti_fundamental_trait_lib::{AntiFundamentalTrait, FundamentalWrapper};

trait Trait1 {}
impl<T: AntiFundamentalTrait> Trait1 for T {}
impl<T> Trait1 for FundamentalWrapper<T> {
    //~^ ERROR conflicting implementations of trait `Trait1` for type `FundamentalWrapper<_>`
    //~| NOTE upstream crates may add a new impl of trait `anti_fundamental_trait_lib::AntiFundamentalTrait` for type `anti_fundamental_trait_lib::FundamentalWrapper<_>` in future versions
}

fn main() {}
