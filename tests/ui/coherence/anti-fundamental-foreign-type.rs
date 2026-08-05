//@ aux-build: anti_fundamental_trait_lib.rs

// Test that `#[rustc_anti_fundamental]` prevents implementing the trait
// on non-local `#[fundamental]` types.

extern crate anti_fundamental_trait_lib;

use anti_fundamental_trait_lib::{
    AntiFundamentalTrait, FundamentalWrapper, NonFundamentalWrapper,
};

struct LocalType;

// OK: implementing on a local type.
impl AntiFundamentalTrait for LocalType {}

// ERROR: implementing on a non-fundamental foreign type wrapping a local type
// (standard orphan check - not covered).
impl AntiFundamentalTrait for NonFundamentalWrapper<LocalType> {}
//~^ ERROR only traits defined in the current crate

// ERROR: implementing on a foreign fundamental type.
impl AntiFundamentalTrait for FundamentalWrapper<LocalType> {}
//~^ ERROR cannot implement `AntiFundamentalTrait` on the fundamental type

// ERROR: implementing on a reference to a foreign fundamental type.
impl AntiFundamentalTrait for &FundamentalWrapper<LocalType> {}
//~^ ERROR cannot implement `AntiFundamentalTrait` on the fundamental type

fn main() {}
