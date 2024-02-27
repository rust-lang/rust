//! This module contains the definitions of most `TypeRelation`s in the type system
//! (except for some relations used for diagnostics and heuristics in the compiler).

pub(super) mod combine;
mod equate;
mod generalize;
mod glb;
mod higher_ranked;
mod lattice;
mod lub;
mod sub;

/// Whether aliases should be related structurally or not. Used
/// to adjust the behavior of generalization and combine.
///
/// This should always be `No` unless in a few special-cases when
/// instantiating canonical responses and in the new solver. Each
/// such case should have a comment explaining why it is used.
#[derive(Debug, Copy, Clone)]
pub enum StructurallyRelateAliases {
    Yes,
    No,
}
