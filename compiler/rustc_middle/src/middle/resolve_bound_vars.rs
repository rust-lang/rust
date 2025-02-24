//! Name resolution for lifetimes and late-bound type and const variables: type declarations.

use rustc_data_structures::sorted_map::SortedMap;
use rustc_errors::ErrorGuaranteed;
use rustc_hir::ItemLocalId;
use rustc_hir::def_id::{DefId, LocalDefId, LocalDefIdMap};
use rustc_macros::{Decodable, Encodable, HashStable, TyDecodable, TyEncodable};

use crate::ty;

#[derive(Clone, Copy, PartialEq, Eq, Hash, TyEncodable, TyDecodable, Debug, HashStable)]
pub enum ResolvedArg {
    StaticLifetime,
    EarlyBound(/* decl */ LocalDefId),
    LateBound(ty::DebruijnIndex, /* late-bound index */ u32, /* decl */ LocalDefId),
    Free(LocalDefId, /* lifetime decl */ LocalDefId),
    Error(ErrorGuaranteed),
}

/// A set containing, at most, one known element.
/// If two distinct values are inserted into a set, then it
/// becomes `Many`, which can be used to detect ambiguities.
#[derive(Copy, Clone, PartialEq, Eq, TyEncodable, TyDecodable, Debug, HashStable)]
pub enum Set1<T> {
    Empty,
    One(T),
    Many,
}

impl<T: PartialEq> Set1<T> {
    pub fn insert(&mut self, value: T) {
        *self = match self {
            Set1::Empty => Set1::One(value),
            Set1::One(old) if *old == value => return,
            _ => Set1::Many,
        };
    }
}

#[derive(Copy, Clone, Debug, HashStable, Encodable, Decodable)]
pub enum ObjectLifetimeDefault {
    Empty,
    Static,
    Ambiguous,
    Param(DefId),
}

/// Maps the id of each bound variable reference to the variable decl
/// that it corresponds to.
#[derive(Debug, Default, HashStable)]
pub struct ResolveBoundVars {
    // Maps from every use of a named (not anonymous) bound var to a
    // `ResolvedArg` describing how that variable is bound.
    pub defs: SortedMap<ItemLocalId, ResolvedArg>,

    // Maps relevant hir items to the bound vars on them. These include:
    // - function defs
    // - function pointers
    // - closures
    // - trait refs
    // - bound types (like `T` in `for<'a> T<'a>: Foo`)
    pub late_bound_vars: SortedMap<ItemLocalId, Vec<ty::BoundVariableKind>>,

    // List captured variables for each opaque type.
    pub opaque_captured_lifetimes: LocalDefIdMap<Vec<(ResolvedArg, LocalDefId)>>,
}
