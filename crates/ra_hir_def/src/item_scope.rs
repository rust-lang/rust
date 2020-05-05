//! Describes items defined or visible (ie, imported) in a certain scope.
//! This is shared between modules and blocks.

use hir_expand::name::Name;
use once_cell::sync::Lazy;
use rustc_hash::FxHashMap;

use crate::{
    per_ns::PerNs, visibility::Visibility, AdtId, BuiltinType, ImplId, MacroDefId, ModuleDefId,
    TraitId,
};

#[derive(Debug, Default, PartialEq, Eq)]
pub struct ItemScope {
    visible: FxHashMap<Name, PerNs>,
    defs: Vec<ModuleDefId>,
    impls: Vec<ImplId>,
    /// Macros visible in current module in legacy textual scope
    ///
    /// For macros invoked by an unqualified identifier like `bar!()`, `legacy_macros` will be searched in first.
    /// If it yields no result, then it turns to module scoped `macros`.
    /// It macros with name qualified with a path like `crate::foo::bar!()`, `legacy_macros` will be skipped,
    /// and only normal scoped `macros` will be searched in.
    ///
    /// Note that this automatically inherit macros defined textually before the definition of module itself.
    ///
    /// Module scoped macros will be inserted into `items` instead of here.
    // FIXME: Macro shadowing in one module is not properly handled. Non-item place macros will
    // be all resolved to the last one defined if shadowing happens.
    legacy_macros: FxHashMap<Name, MacroDefId>,
}

pub(crate) static BUILTIN_SCOPE: Lazy<FxHashMap<Name, PerNs>> = Lazy::new(|| {
    BuiltinType::ALL
        .iter()
        .map(|(name, ty)| (name.clone(), PerNs::types(ty.clone().into(), Visibility::Public)))
        .collect()
});

/// Shadow mode for builtin type which can be shadowed by module.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum BuiltinShadowMode {
    /// Prefer user-defined modules (or other types) over builtins.
    Module,
    /// Prefer builtins over user-defined modules (but not other types).
    Other,
}

/// Legacy macros can only be accessed through special methods like `get_legacy_macros`.
/// Other methods will only resolve values, types and module scoped macros only.
impl ItemScope {
    pub fn entries<'a>(&'a self) -> impl Iterator<Item = (&'a Name, PerNs)> + 'a {
        //FIXME: shadowing
        self.visible.iter().map(|(n, def)| (n, *def))
    }

    pub fn entries_without_primitives<'a>(
        &'a self,
    ) -> impl Iterator<Item = (&'a Name, PerNs)> + 'a {
        self.visible.iter().map(|(n, def)| (n, *def))
    }

    pub fn declarations(&self) -> impl Iterator<Item = ModuleDefId> + '_ {
        self.defs.iter().copied()
    }

    pub fn impls(&self) -> impl Iterator<Item = ImplId> + ExactSizeIterator + '_ {
        self.impls.iter().copied()
    }

    pub fn visibility_of(&self, def: ModuleDefId) -> Option<Visibility> {
        self.name_of(ItemInNs::Types(def))
            .or_else(|| self.name_of(ItemInNs::Values(def)))
            .map(|(_, v)| v)
    }

    /// Iterate over all module scoped macros
    pub(crate) fn macros<'a>(&'a self) -> impl Iterator<Item = (&'a Name, MacroDefId)> + 'a {
        self.visible.iter().filter_map(|(name, def)| def.take_macros().map(|macro_| (name, macro_)))
    }

    /// Iterate over all legacy textual scoped macros visible at the end of the module
    pub(crate) fn legacy_macros<'a>(&'a self) -> impl Iterator<Item = (&'a Name, MacroDefId)> + 'a {
        self.legacy_macros.iter().map(|(name, def)| (name, *def))
    }

    /// Get a name from current module scope, legacy macros are not included
    pub(crate) fn get(&self, name: &Name) -> PerNs {
        self.visible.get(name).copied().unwrap_or_else(PerNs::none)
    }

    pub(crate) fn name_of(&self, item: ItemInNs) -> Option<(&Name, Visibility)> {
        for (name, per_ns) in &self.visible {
            if let Some(vis) = item.match_with(*per_ns) {
                return Some((name, vis));
            }
        }
        None
    }

    pub(crate) fn traits<'a>(&'a self) -> impl Iterator<Item = TraitId> + 'a {
        self.visible.values().filter_map(|def| match def.take_types() {
            Some(ModuleDefId::TraitId(t)) => Some(t),
            _ => None,
        })
    }

    pub(crate) fn define_def(&mut self, def: ModuleDefId) {
        self.defs.push(def)
    }

    pub(crate) fn get_legacy_macro(&self, name: &Name) -> Option<MacroDefId> {
        self.legacy_macros.get(name).copied()
    }

    pub(crate) fn define_impl(&mut self, imp: ImplId) {
        self.impls.push(imp)
    }

    pub(crate) fn define_legacy_macro(&mut self, name: Name, mac: MacroDefId) {
        self.legacy_macros.insert(name, mac);
    }

    pub(crate) fn push_res(&mut self, name: Name, def: PerNs) -> bool {
        let mut changed = false;
        let existing = self.visible.entry(name).or_default();

        if existing.types.is_none() && def.types.is_some() {
            existing.types = def.types;
            changed = true;
        }
        if existing.values.is_none() && def.values.is_some() {
            existing.values = def.values;
            changed = true;
        }
        if existing.macros.is_none() && def.macros.is_some() {
            existing.macros = def.macros;
            changed = true;
        }

        changed
    }

    pub(crate) fn resolutions<'a>(&'a self) -> impl Iterator<Item = (Name, PerNs)> + 'a {
        self.visible.iter().map(|(name, res)| (name.clone(), *res))
    }

    pub(crate) fn collect_legacy_macros(&self) -> FxHashMap<Name, MacroDefId> {
        self.legacy_macros.clone()
    }
}

impl PerNs {
    pub(crate) fn from_def(def: ModuleDefId, v: Visibility, has_constructor: bool) -> PerNs {
        match def {
            ModuleDefId::ModuleId(_) => PerNs::types(def, v),
            ModuleDefId::FunctionId(_) => PerNs::values(def, v),
            ModuleDefId::AdtId(adt) => match adt {
                AdtId::UnionId(_) => PerNs::types(def, v),
                AdtId::EnumId(_) => PerNs::types(def, v),
                AdtId::StructId(_) => {
                    if has_constructor {
                        PerNs::both(def, def, v)
                    } else {
                        PerNs::types(def, v)
                    }
                }
            },
            ModuleDefId::EnumVariantId(_) => PerNs::both(def, def, v),
            ModuleDefId::ConstId(_) | ModuleDefId::StaticId(_) => PerNs::values(def, v),
            ModuleDefId::TraitId(_) => PerNs::types(def, v),
            ModuleDefId::TypeAliasId(_) => PerNs::types(def, v),
            ModuleDefId::BuiltinType(_) => PerNs::types(def, v),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum ItemInNs {
    Types(ModuleDefId),
    Values(ModuleDefId),
    Macros(MacroDefId),
}

impl ItemInNs {
    fn match_with(self, per_ns: PerNs) -> Option<Visibility> {
        match self {
            ItemInNs::Types(def) => {
                per_ns.types.filter(|(other_def, _)| *other_def == def).map(|(_, vis)| vis)
            }
            ItemInNs::Values(def) => {
                per_ns.values.filter(|(other_def, _)| *other_def == def).map(|(_, vis)| vis)
            }
            ItemInNs::Macros(def) => {
                per_ns.macros.filter(|(other_def, _)| *other_def == def).map(|(_, vis)| vis)
            }
        }
    }

    pub fn as_module_def_id(self) -> Option<ModuleDefId> {
        match self {
            ItemInNs::Types(id) | ItemInNs::Values(id) => Some(id),
            ItemInNs::Macros(_) => None,
        }
    }
}
