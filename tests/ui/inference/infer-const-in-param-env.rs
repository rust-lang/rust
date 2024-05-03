//! Issue #119381: encountering `ty::ConstKind::Infer(InferConst::Var(_))` inside a `ParamEnv`

#![feature(with_negative_coherence)]
trait Trait {}
impl<const N: u8> Trait for [(); N] {}
impl<const N: i8> Trait for [(); N] {}
//~^ conflicting implementations of trait `Trait` for type `[(); _]`
//~| mismatched types
//~^^^^ mismatched types

fn main() {}
