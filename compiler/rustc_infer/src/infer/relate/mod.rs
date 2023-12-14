//! This module contains the definitions of most `TypeRelation`s in the type system
//! (except for the NLL `TypeRelating`, and some relations used for diagnostics
//! and heuristics in the compiler).

pub(super) mod combine;
pub(super) mod equate;
pub(super) mod generalize;
pub(super) mod glb;
pub(super) mod lattice;
pub(super) mod lub;
pub(super) mod sub;
