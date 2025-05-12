#![crate_name = "user"]
//@ aux-crate:default_generic_args=default-generic-args.rs
//@ edition:2021

//@ has user/type.BoxedStr.html
//@ has - '//*[@class="rust item-decl"]//code' "Box<str>"
pub use default_generic_args::BoxedStr;

//@ has user/type.IntMap.html
//@ has - '//*[@class="rust item-decl"]//code' "HashMap<i64, u64>"
pub use default_generic_args::IntMap;

//@ has user/type.T0.html
//@ has - '//*[@class="rust item-decl"]//code' "TyPair<i32>"
pub use default_generic_args::T0;

//@ has user/type.T1.html
//@ has - '//*[@class="rust item-decl"]//code' "TyPair<i32, u32>"
pub use default_generic_args::T1;

//@ has user/type.T2.html
//@ has - '//*[@class="rust item-decl"]//code' "TyPair<i32, K>"
pub use default_generic_args::T2;

//@ has user/type.T3.html
//@ has - '//*[@class="rust item-decl"]//code' "TyPair<Q>"
pub use default_generic_args::T3;

//@ has user/type.C0.html
//@ has - '//*[@class="rust item-decl"]//code' "CtPair<43>"
pub use default_generic_args::C0;

//@ has user/type.C1.html
//@ has - '//*[@class="rust item-decl"]//code' "CtPair<0, 1>"
pub use default_generic_args::C1;

//@ has user/type.C2.html
//@ has - '//*[@class="rust item-decl"]//code' "CtPair<default_generic_args::::C2::{constant#0}, 3>"
pub use default_generic_args::C2;

//@ has user/type.R0.html
//@ has - '//*[@class="rust item-decl"]//code' "Re<'q>"
pub use default_generic_args::R0;

//@ has user/type.R1.html
//@ has - '//*[@class="rust item-decl"]//code' "Re<'q>"
pub use default_generic_args::R1;

//@ has user/type.R2.html
// Check that we consider regions:
//@ has - '//*[@class="rust item-decl"]//code' "Re<'q, &'static ()>"
pub use default_generic_args::R2;

//@ has user/type.H0.html
// Check that we handle higher-ranked regions correctly:
//@ has - '//*[@class="rust item-decl"]//code' "fn(for<'a> fn(Re<'a>))"
pub use default_generic_args::H0;

//@ has user/type.H1.html
// Check that we don't conflate distinct universially quantified regions (#1):
//@ has - '//*[@class="rust item-decl"]//code' "for<'b> fn(for<'a> fn(Re<'a, &'b ()>))"
pub use default_generic_args::H1;

//@ has user/type.H2.html
// Check that we don't conflate distinct universially quantified regions (#2):
//@ has - '//*[@class="rust item-decl"]//code' "for<'a> fn(for<'b> fn(Re<'a, &'b ()>))"
pub use default_generic_args::H2;

//@ has user/type.P0.html
//@ has - '//*[@class="rust item-decl"]//code' "Proj<()>"
pub use default_generic_args::P0;

//@ has user/type.P1.html
//@ has - '//*[@class="rust item-decl"]//code' "Proj<(), bool>"
pub use default_generic_args::P1;

//@ has user/type.P2.html
//@ has - '//*[@class="rust item-decl"]//code' "Proj<(), ()>"
pub use default_generic_args::P2;

//@ has user/type.A0.html
//@ has - '//*[@class="rust item-decl"]//code' "Alpha;"
pub use default_generic_args::A0;

//@ has user/type.A1.html
// Demonstrates that we currently don't elide generic arguments that are alpha-equivalent to their
// respective generic parameter (after instantiation) for perf reasons (it would require us to
// create an inference context).
//@ has - '//*[@class="rust item-decl"]//code' "Alpha<for<'arbitrary> fn(&'arbitrary ())>"
pub use default_generic_args::A1;

//@ has user/type.M0.html
// Test that we don't elide `u64` even if it coincides with `A`'s default precisely because
// `()` is not the default of `B`. Mindlessly eliding `u64` would lead to `M<()>` which is a
// different type (`M<(), u64>` versus `M<u64, ()>`).
//@ has - '//*[@class="rust item-decl"]//code' "Multi<u64, ()>"
pub use default_generic_args::M0;

//@ has user/type.D0.html
//@ has - '//*[@class="rust item-decl"]//code' "dyn for<'a> Trait0<'a>"
pub use default_generic_args::D0;

// Regression test for issue #119529.
// Check that we correctly elide def ty&const args inside trait object types.

//@ has user/type.D1.html
//@ has - '//*[@class="rust item-decl"]//code' "dyn Trait1<T>"
pub use default_generic_args::D1;
//@ has user/type.D2.html
//@ has - '//*[@class="rust item-decl"]//code' "dyn Trait1<(), K>"
pub use default_generic_args::D2;
//@ has user/type.D3.html
//@ has - '//*[@class="rust item-decl"]//code' "dyn Trait1;"
pub use default_generic_args::D3;
