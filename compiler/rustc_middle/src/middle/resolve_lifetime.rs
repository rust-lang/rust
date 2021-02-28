//! Name resolution for lifetimes: type declarations.

use crate::ty;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::{GenericParam, ItemLocalId};
use rustc_hir::{GenericParamKind, LifetimeParamKind};
use rustc_macros::HashStable;

/// The origin of a named lifetime definition.
///
/// This is used to prevent the usage of in-band lifetimes in `Fn`/`fn` syntax.
#[derive(Copy, Clone, PartialEq, Eq, Hash, TyEncodable, TyDecodable, Debug, HashStable)]
pub enum LifetimeDefOrigin {
    // Explicit binders like `fn foo<'a>(x: &'a u8)` or elided like `impl Foo<&u32>`
    ExplicitOrElided,
    // In-band declarations like `fn foo(x: &'a u8)`
    InBand,
    // Some kind of erroneous origin
    Error,
}

impl LifetimeDefOrigin {
    pub fn from_param(param: &GenericParam<'_>) -> Self {
        match param.kind {
            GenericParamKind::Lifetime { kind } => match kind {
                LifetimeParamKind::InBand => LifetimeDefOrigin::InBand,
                LifetimeParamKind::Explicit => LifetimeDefOrigin::ExplicitOrElided,
                LifetimeParamKind::Elided => LifetimeDefOrigin::ExplicitOrElided,
                LifetimeParamKind::Error => LifetimeDefOrigin::Error,
            },
            _ => bug!("expected a lifetime param"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, TyEncodable, TyDecodable, Debug, HashStable)]
pub enum Region {
    Static,
    EarlyBound(/* index */ u32, /* lifetime decl */ DefId, LifetimeDefOrigin),
    LateBound(ty::DebruijnIndex, /* lifetime decl */ DefId, LifetimeDefOrigin),
    LateBoundAnon(ty::DebruijnIndex, /* anon index */ u32),
    Free(DefId, /* lifetime decl */ DefId),
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
}
