//! In rust, it is possible to have a value, a type and a macro with the same
//! name without conflicts.
//!
//! `PerNs` (per namespace) captures this.

use bitflags::bitflags;

use crate::{
    item_scope::{ImportId, ImportOrExternCrate, ItemInNs},
    visibility::Visibility,
    MacroId, ModuleDefId,
};

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub enum Namespace {
    Types,
    Values,
    Macros,
}

bitflags! {
    /// Describes only the presence/absence of each namespace, without its value.
    #[derive(Debug, PartialEq, Eq)]
    pub(crate) struct NsAvailability : u32 {
        const TYPES = 1 << 0;
        const VALUES = 1 << 1;
        const MACROS = 1 << 2;
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct PerNs {
    pub types: Option<(ModuleDefId, Visibility, Option<ImportOrExternCrate>)>,
    pub values: Option<(ModuleDefId, Visibility, Option<ImportId>)>,
    pub macros: Option<(MacroId, Visibility, Option<ImportId>)>,
}

impl PerNs {
    pub(crate) fn availability(&self) -> NsAvailability {
        let mut result = NsAvailability::empty();
        result.set(NsAvailability::TYPES, self.types.is_some());
        result.set(NsAvailability::VALUES, self.values.is_some());
        result.set(NsAvailability::MACROS, self.macros.is_some());
        result
    }

    pub fn none() -> PerNs {
        PerNs { types: None, values: None, macros: None }
    }

    pub fn values(t: ModuleDefId, v: Visibility, i: Option<ImportId>) -> PerNs {
        PerNs { types: None, values: Some((t, v, i)), macros: None }
    }

    pub fn types(t: ModuleDefId, v: Visibility, i: Option<ImportOrExternCrate>) -> PerNs {
        PerNs { types: Some((t, v, i)), values: None, macros: None }
    }

    pub fn both(
        types: ModuleDefId,
        values: ModuleDefId,
        v: Visibility,
        i: Option<ImportOrExternCrate>,
    ) -> PerNs {
        PerNs {
            types: Some((types, v, i)),
            values: Some((values, v, i.and_then(ImportOrExternCrate::into_import))),
            macros: None,
        }
    }

    pub fn macros(macro_: MacroId, v: Visibility, i: Option<ImportId>) -> PerNs {
        PerNs { types: None, values: None, macros: Some((macro_, v, i)) }
    }

    pub fn is_none(&self) -> bool {
        self.types.is_none() && self.values.is_none() && self.macros.is_none()
    }

    pub fn is_full(&self) -> bool {
        self.types.is_some() && self.values.is_some() && self.macros.is_some()
    }

    pub fn take_types(self) -> Option<ModuleDefId> {
        self.types.map(|it| it.0)
    }

    pub fn take_types_full(self) -> Option<(ModuleDefId, Visibility, Option<ImportOrExternCrate>)> {
        self.types
    }

    pub fn take_values(self) -> Option<ModuleDefId> {
        self.values.map(|it| it.0)
    }

    pub fn take_values_import(self) -> Option<(ModuleDefId, Option<ImportId>)> {
        self.values.map(|it| (it.0, it.2))
    }

    pub fn take_macros(self) -> Option<MacroId> {
        self.macros.map(|it| it.0)
    }

    pub fn take_macros_import(self) -> Option<(MacroId, Option<ImportId>)> {
        self.macros.map(|it| (it.0, it.2))
    }

    pub fn filter_visibility(self, mut f: impl FnMut(Visibility) -> bool) -> PerNs {
        let _p = tracing::info_span!("PerNs::filter_visibility").entered();
        PerNs {
            types: self.types.filter(|&(_, v, _)| f(v)),
            values: self.values.filter(|&(_, v, _)| f(v)),
            macros: self.macros.filter(|&(_, v, _)| f(v)),
        }
    }

    pub fn with_visibility(self, vis: Visibility) -> PerNs {
        PerNs {
            types: self.types.map(|(it, _, c)| (it, vis, c)),
            values: self.values.map(|(it, _, c)| (it, vis, c)),
            macros: self.macros.map(|(it, _, import)| (it, vis, import)),
        }
    }

    pub fn or(self, other: PerNs) -> PerNs {
        PerNs {
            types: self.types.or(other.types),
            values: self.values.or(other.values),
            macros: self.macros.or(other.macros),
        }
    }

    pub fn or_else(self, f: impl FnOnce() -> PerNs) -> PerNs {
        if self.is_full() {
            self
        } else {
            self.or(f())
        }
    }

    pub fn iter_items(self) -> impl Iterator<Item = (ItemInNs, Option<ImportOrExternCrate>)> {
        let _p = tracing::info_span!("PerNs::iter_items").entered();
        self.types
            .map(|it| (ItemInNs::Types(it.0), it.2))
            .into_iter()
            .chain(
                self.values
                    .map(|it| (ItemInNs::Values(it.0), it.2.map(ImportOrExternCrate::Import))),
            )
            .chain(
                self.macros
                    .map(|it| (ItemInNs::Macros(it.0), it.2.map(ImportOrExternCrate::Import))),
            )
    }
}
