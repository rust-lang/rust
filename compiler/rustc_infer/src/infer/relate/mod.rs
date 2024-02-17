//! This module contains the definitions of most `TypeRelation`s in the type system
//! (except for some relations used for diagnostics and heuristics in the compiler).

pub(super) mod combine;
mod equate;
mod generalize;
mod glb;
mod higher_ranked;
mod lattice;
mod lub;
pub mod nll;
mod sub;
