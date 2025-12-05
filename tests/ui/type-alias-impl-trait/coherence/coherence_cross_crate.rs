//@ aux-build: coherence_cross_crate_trait_decl.rs
// This test ensures that adding an `impl SomeTrait for i32` within
// `coherence_cross_crate_trait_decl` is not a breaking change, by
// making sure that even without such an impl this test fails to compile.

#![feature(type_alias_impl_trait)]

extern crate coherence_cross_crate_trait_decl;

use coherence_cross_crate_trait_decl::SomeTrait;

trait OtherTrait {}

type Alias = impl SomeTrait;

#[define_opaque(Alias)]
fn constrain() -> Alias {
    ()
}

impl OtherTrait for Alias {}
impl OtherTrait for i32 {}
//~^ ERROR: conflicting implementations of trait `OtherTrait` for type `Alias`

fn main() {}
