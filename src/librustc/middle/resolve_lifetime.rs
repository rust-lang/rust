//! Name resolution for lifetimes: type declarations.

use crate::hir::def_id::{DefId, LocalDefId};
use crate::hir::{GenericParam, ItemLocalId};
use crate::hir::{GenericParamKind, LifetimeParamKind};
use crate::ty;

use crate::util::nodemap::{FxHashMap, FxHashSet, HirIdMap, HirIdSet};
use rustc_macros::HashStable;

/// The origin of a named lifetime definition.
///
/// This is used to prevent the usage of in-band lifetimes in `Fn`/`fn` syntax.
#[derive(Copy, Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable, Debug, HashStable)]
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

#[derive(Clone, Copy, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable, Debug, HashStable)]
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
#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Debug, HashStable)]
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
#[derive(HashStable)]
pub struct ResolveLifetimes {
    defs: FxHashMap<LocalDefId, FxHashMap<ItemLocalId, Region>>,
    late_bound: FxHashMap<LocalDefId, FxHashSet<ItemLocalId>>,
    object_lifetime_defaults:
        FxHashMap<LocalDefId, FxHashMap<ItemLocalId, Vec<ObjectLifetimeDefault>>>,
}

impl ResolveLifetimes {
    pub fn new(
        defs: HirIdMap<Region>,
        late_bound: HirIdSet,
        object_lifetime_defaults: HirIdMap<Vec<ObjectLifetimeDefault>>,
    ) -> Self {
        let defs = {
            let mut map = FxHashMap::<_, FxHashMap<_, _>>::default();
            for (hir_id, v) in defs {
                let map = map.entry(hir_id.owner_local_def_id()).or_default();
                map.insert(hir_id.local_id, v);
            }
            map
        };
        let late_bound = {
            let mut map = FxHashMap::<_, FxHashSet<_>>::default();
            for hir_id in late_bound {
                let map = map.entry(hir_id.owner_local_def_id()).or_default();
                map.insert(hir_id.local_id);
            }
            map
        };
        let object_lifetime_defaults = {
            let mut map = FxHashMap::<_, FxHashMap<_, _>>::default();
            for (hir_id, v) in object_lifetime_defaults {
                let map = map.entry(hir_id.owner_local_def_id()).or_default();
                map.insert(hir_id.local_id, v);
            }
            map
        };

        Self { defs, late_bound, object_lifetime_defaults }
    }

    pub fn named_region_map(&self, id: &LocalDefId) -> Option<&FxHashMap<ItemLocalId, Region>> {
        self.defs.get(id)
    }

    pub fn is_late_bound_map(&self, id: &LocalDefId) -> Option<&FxHashSet<ItemLocalId>> {
        self.late_bound.get(id)
    }

    pub fn object_lifetime_defaults_map(
        &self,
        id: &LocalDefId,
    ) -> Option<&FxHashMap<ItemLocalId, Vec<ObjectLifetimeDefault>>> {
        self.object_lifetime_defaults.get(id)
    }
}
