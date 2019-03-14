//! Name resolution algorithm. The end result of the algorithm is an `ItemMap`:
//! a map which maps each module to its scope: the set of items visible in the
//! module. That is, we only resolve imports here, name resolution of item
//! bodies will be done in a separate step.
//!
//! Like Rustc, we use an interactive per-crate algorithm: we start with scopes
//! containing only directly defined items, and then iteratively resolve
//! imports.
//!
//! To make this work nicely in the IDE scenario, we place `InputModuleItems`
//! in between raw syntax and name resolution. `InputModuleItems` are computed
//! using only the module's syntax, and it is all directly defined items plus
//! imports. The plan is to make `InputModuleItems` independent of local
//! modifications (that is, typing inside a function should not change IMIs),
//! so that the results of name resolution can be preserved unless the module
//! structure itself is modified.
pub(crate) mod lower;
pub(crate) mod crate_def_map;

use rustc_hash::FxHashMap;
use ra_db::Edition;

use crate::{
    ModuleDef, Name,
    nameres::lower::ImportId,
};

pub(crate) use self::crate_def_map::{CrateDefMap, ModuleId};

#[derive(Debug, Default, PartialEq, Eq, Clone)]
pub struct ModuleScope {
    pub(crate) items: FxHashMap<Name, Resolution>,
}

impl ModuleScope {
    pub fn entries<'a>(&'a self) -> impl Iterator<Item = (&'a Name, &'a Resolution)> + 'a {
        self.items.iter()
    }
    pub fn get(&self, name: &Name) -> Option<&Resolution> {
        self.items.get(name)
    }
}

/// `Resolution` is basically `DefId` atm, but it should account for stuff like
/// multiple namespaces, ambiguity and errors.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Resolution {
    /// None for unresolved
    pub def: PerNs<ModuleDef>,
    /// ident by which this is imported into local scope.
    pub import: Option<ImportId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Namespace {
    Types,
    Values,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct PerNs<T> {
    pub types: Option<T>,
    pub values: Option<T>,
}

impl<T> Default for PerNs<T> {
    fn default() -> Self {
        PerNs { types: None, values: None }
    }
}

impl<T> PerNs<T> {
    pub fn none() -> PerNs<T> {
        PerNs { types: None, values: None }
    }

    pub fn values(t: T) -> PerNs<T> {
        PerNs { types: None, values: Some(t) }
    }

    pub fn types(t: T) -> PerNs<T> {
        PerNs { types: Some(t), values: None }
    }

    pub fn both(types: T, values: T) -> PerNs<T> {
        PerNs { types: Some(types), values: Some(values) }
    }

    pub fn is_none(&self) -> bool {
        self.types.is_none() && self.values.is_none()
    }

    pub fn is_both(&self) -> bool {
        self.types.is_some() && self.values.is_some()
    }

    pub fn take(self, namespace: Namespace) -> Option<T> {
        match namespace {
            Namespace::Types => self.types,
            Namespace::Values => self.values,
        }
    }

    pub fn take_types(self) -> Option<T> {
        self.take(Namespace::Types)
    }

    pub fn take_values(self) -> Option<T> {
        self.take(Namespace::Values)
    }

    pub fn get(&self, namespace: Namespace) -> Option<&T> {
        self.as_ref().take(namespace)
    }

    pub fn as_ref(&self) -> PerNs<&T> {
        PerNs { types: self.types.as_ref(), values: self.values.as_ref() }
    }

    pub fn or(self, other: PerNs<T>) -> PerNs<T> {
        PerNs { types: self.types.or(other.types), values: self.values.or(other.values) }
    }

    pub fn and_then<U>(self, f: impl Fn(T) -> Option<U>) -> PerNs<U> {
        PerNs { types: self.types.and_then(&f), values: self.values.and_then(&f) }
    }

    pub fn map<U>(self, f: impl Fn(T) -> U) -> PerNs<U> {
        PerNs { types: self.types.map(&f), values: self.values.map(&f) }
    }
}

#[derive(Debug, Clone)]
struct ResolvePathResult {
    resolved_def: PerNs<ModuleDef>,
    segment_index: Option<usize>,
    reached_fixedpoint: ReachedFixedPoint,
}

impl ResolvePathResult {
    fn empty(reached_fixedpoint: ReachedFixedPoint) -> ResolvePathResult {
        ResolvePathResult::with(PerNs::none(), reached_fixedpoint, None)
    }

    fn with(
        resolved_def: PerNs<ModuleDef>,
        reached_fixedpoint: ReachedFixedPoint,
        segment_index: Option<usize>,
    ) -> ResolvePathResult {
        ResolvePathResult { resolved_def, reached_fixedpoint, segment_index }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolveMode {
    Import,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReachedFixedPoint {
    Yes,
    No,
}
