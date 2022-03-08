//! Name resolution for lifetimes: type declarations.

use crate::ty;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::ItemLocalId;
use rustc_macros::HashStable;

#[derive(Clone, Copy, PartialEq, Eq, Hash, TyEncodable, TyDecodable, Debug, HashStable)]
pub enum Region {
    Static,
    EarlyBound(/* index */ u32, /* lifetime decl */ DefId),
    LateBound(ty::DebruijnIndex, /* late-bound index */ u32, /* lifetime decl */ DefId),
    LateBoundAnon(ty::DebruijnIndex, /* late-bound index */ u32, /* anon index */ u32),
    Free(DefId, /* lifetime decl */ DefId),
}

/// This is used in diagnostics to improve suggestions for missing generic arguments.
/// It gives information on the type of lifetimes that are in scope for a particular `PathSegment`,
/// so that we can e.g. suggest elided-lifetimes-in-paths of the form <'_, '_> e.g.
#[derive(Clone, PartialEq, Eq, Hash, TyEncodable, TyDecodable, Debug, HashStable)]
pub enum LifetimeScopeForPath {
    // Contains all lifetime names that are in scope and could possibly be used in generics
    // arguments of path.
    NonElided(Vec<String>),

    // Information that allows us to suggest args of the form `<'_>` in case
    // no generic arguments were provided for a path.
    Elided,
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

pub type ObjectLifetimeDefault = Set1<Region>;

/// Maps the id of each lifetime reference to the lifetime decl
/// that it corresponds to.
#[derive(Default, HashStable, Debug)]
pub struct ResolveLifetimes {
    /// Maps from every use of a named (not anonymous) lifetime to a
    /// `Region` describing how that region is bound
    pub defs: FxHashMap<LocalDefId, FxHashMap<ItemLocalId, Region>>,

    /// Set of lifetime def ids that are late-bound; a region can
    /// be late-bound if (a) it does NOT appear in a where-clause and
    /// (b) it DOES appear in the arguments.
    pub late_bound: FxHashMap<LocalDefId, FxHashSet<ItemLocalId>>,

    pub late_bound_vars: FxHashMap<LocalDefId, FxHashMap<ItemLocalId, Vec<ty::BoundVariableKind>>>,
}
