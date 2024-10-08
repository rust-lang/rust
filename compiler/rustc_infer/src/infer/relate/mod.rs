//! This module contains the definitions of most `TypeRelation`s in the type system
//! (except for some relations used for diagnostics and heuristics in the compiler).
//! As well as the implementation of `Relate` for interned things (`Ty`/`Const`/etc).

pub use rustc_middle::ty::relate::RelateResult;
pub use rustc_next_trait_solver::relate::*;

pub use self::combine::PredicateEmittingRelation;

#[allow(hidden_glob_reexports)]
pub(super) mod combine;
mod generalize;
mod higher_ranked;
pub(super) mod lattice;
pub(super) mod type_relating;
