//! In rust, it is possible to have a value, a type and a macro with the same
//! name without conflicts.
//!
//! `PerNs` (per namespace) captures this.

use bitflags::bitflags;

use crate::{
    MacroId, ModuleDefId,
    item_scope::{ImportId, ImportOrExternCrate, ImportOrGlob, ItemInNs},
    visibility::Visibility,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Item<Def, Import = ImportId> {
    pub def: Def,
    pub vis: Visibility,
    pub import: Option<Import>,
}

pub type TypesItem = Item<ModuleDefId, ImportOrExternCrate>;
pub type ValuesItem = Item<ModuleDefId, ImportOrGlob>;
// May be Externcrate for `[macro_use]`'d macros
pub type MacrosItem = Item<MacroId, ImportOrExternCrate>;

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct PerNs {
    pub types: Option<TypesItem>,
    pub values: Option<ValuesItem>,
    pub macros: Option<MacrosItem>,
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

    pub fn values(def: ModuleDefId, vis: Visibility, import: Option<ImportOrGlob>) -> PerNs {
        PerNs { types: None, values: Some(Item { def, vis, import }), macros: None }
    }

    pub fn types(def: ModuleDefId, vis: Visibility, import: Option<ImportOrExternCrate>) -> PerNs {
        PerNs { types: Some(Item { def, vis, import }), values: None, macros: None }
    }

    pub fn both(
        types: ModuleDefId,
        values: ModuleDefId,
        vis: Visibility,
        import: Option<ImportOrExternCrate>,
    ) -> PerNs {
        PerNs {
            types: Some(Item { def: types, vis, import }),
            values: Some(Item {
                def: values,
                vis,
                import: import.and_then(ImportOrExternCrate::import_or_glob),
            }),
            macros: None,
        }
    }

    pub fn macros(def: MacroId, vis: Visibility, import: Option<ImportOrExternCrate>) -> PerNs {
        PerNs { types: None, values: None, macros: Some(Item { def, vis, import }) }
    }

    pub fn is_none(&self) -> bool {
        self.types.is_none() && self.values.is_none() && self.macros.is_none()
    }

    pub fn is_full(&self) -> bool {
        self.types.is_some() && self.values.is_some() && self.macros.is_some()
    }

    pub fn take_types(self) -> Option<ModuleDefId> {
        self.types.map(|it| it.def)
    }

    pub fn take_types_full(self) -> Option<TypesItem> {
        self.types
    }

    pub fn take_values(self) -> Option<ModuleDefId> {
        self.values.map(|it| it.def)
    }

    pub fn take_values_import(self) -> Option<(ModuleDefId, Option<ImportOrGlob>)> {
        self.values.map(|it| (it.def, it.import))
    }

    pub fn take_macros(self) -> Option<MacroId> {
        self.macros.map(|it| it.def)
    }

    pub fn take_macros_import(self) -> Option<(MacroId, Option<ImportOrExternCrate>)> {
        self.macros.map(|it| (it.def, it.import))
    }

    pub fn filter_visibility(self, mut f: impl FnMut(Visibility) -> bool) -> PerNs {
        let _p = tracing::info_span!("PerNs::filter_visibility").entered();
        PerNs {
            types: self.types.filter(|def| f(def.vis)),
            values: self.values.filter(|def| f(def.vis)),
            macros: self.macros.filter(|def| f(def.vis)),
        }
    }

    pub fn with_visibility(self, vis: Visibility) -> PerNs {
        PerNs {
            types: self.types.map(|def| Item { vis, ..def }),
            values: self.values.map(|def| Item { vis, ..def }),
            macros: self.macros.map(|def| Item { vis, ..def }),
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
        if self.is_full() { self } else { self.or(f()) }
    }

    pub fn iter_items(self) -> impl Iterator<Item = (ItemInNs, Option<ImportOrExternCrate>)> {
        let _p = tracing::info_span!("PerNs::iter_items").entered();
        self.types
            .map(|it| (ItemInNs::Types(it.def), it.import))
            .into_iter()
            .chain(
                self.values
                    .map(|it| (ItemInNs::Values(it.def), it.import.map(ImportOrExternCrate::from))),
            )
            .chain(self.macros.map(|it| (ItemInNs::Macros(it.def), it.import)))
    }
}
