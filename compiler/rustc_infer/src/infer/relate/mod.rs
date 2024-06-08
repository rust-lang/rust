//! This module contains the definitions of most `TypeRelation`s in the type system
//! (except for some relations used for diagnostics and heuristics in the compiler).
//! As well as the implementation of `Relate` for interned things (`Ty`/`Const`/etc).

pub use rustc_middle::ty::relate::*;

pub use self::_match::MatchAgainstFreshVars;
pub use self::combine::CombineFields;
pub use self::combine::ObligationEmittingRelation;

pub mod _match;
pub(super) mod combine;
mod generalize;
mod glb;
mod higher_ranked;
mod lattice;
mod lub;
mod type_relating;
