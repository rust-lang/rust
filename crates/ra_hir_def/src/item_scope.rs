//! Describes items defined or visible (ie, imported) in a certain scope.
//! This is shared between modules and blocks.

use hir_expand::name::Name;
use once_cell::sync::Lazy;
use rustc_hash::FxHashMap;

use crate::{per_ns::PerNs, AdtId, BuiltinType, ImplId, MacroDefId, ModuleDefId, TraitId};

#[derive(Debug, Default, PartialEq, Eq)]
pub struct ItemScope {
    visible: FxHashMap<Name, Resolution>,
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

static BUILTIN_SCOPE: Lazy<FxHashMap<Name, Resolution>> = Lazy::new(|| {
    BuiltinType::ALL
        .iter()
        .map(|(name, ty)| {
            (name.clone(), Resolution { def: PerNs::types(ty.clone().into()), import: false })
        })
        .collect()
});

/// Shadow mode for builtin type which can be shadowed by module.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum BuiltinShadowMode {
    // Prefer Module
    Module,
    // Prefer Other Types
    Other,
}

/// Legacy macros can only be accessed through special methods like `get_legacy_macros`.
/// Other methods will only resolve values, types and module scoped macros only.
impl ItemScope {
    pub fn entries<'a>(&'a self) -> impl Iterator<Item = (&'a Name, &'a Resolution)> + 'a {
        //FIXME: shadowing
        self.visible.iter().chain(BUILTIN_SCOPE.iter())
    }

    pub fn declarations(&self) -> impl Iterator<Item = ModuleDefId> + '_ {
        self.entries()
            .filter_map(|(_name, res)| if !res.import { Some(res.def) } else { None })
            .flat_map(|per_ns| {
                per_ns.take_types().into_iter().chain(per_ns.take_values().into_iter())
            })
    }

    pub fn impls(&self) -> impl Iterator<Item = ImplId> + ExactSizeIterator + '_ {
        self.impls.iter().copied()
    }

    /// Iterate over all module scoped macros
    pub(crate) fn macros<'a>(&'a self) -> impl Iterator<Item = (&'a Name, MacroDefId)> + 'a {
        self.visible
            .iter()
            .filter_map(|(name, res)| res.def.take_macros().map(|macro_| (name, macro_)))
    }

    /// Iterate over all legacy textual scoped macros visible at the end of the module
    pub(crate) fn legacy_macros<'a>(&'a self) -> impl Iterator<Item = (&'a Name, MacroDefId)> + 'a {
        self.legacy_macros.iter().map(|(name, def)| (name, *def))
    }

    /// Get a name from current module scope, legacy macros are not included
    pub(crate) fn get(&self, name: &Name, shadow: BuiltinShadowMode) -> Option<&Resolution> {
        match shadow {
            BuiltinShadowMode::Module => self.visible.get(name).or_else(|| BUILTIN_SCOPE.get(name)),
            BuiltinShadowMode::Other => {
                let item = self.visible.get(name);
                if let Some(res) = item {
                    if let Some(ModuleDefId::ModuleId(_)) = res.def.take_types() {
                        return BUILTIN_SCOPE.get(name).or(item);
                    }
                }

                item.or_else(|| BUILTIN_SCOPE.get(name))
            }
        }
    }

    pub(crate) fn traits<'a>(&'a self) -> impl Iterator<Item = TraitId> + 'a {
        self.visible.values().filter_map(|r| match r.def.take_types() {
            Some(ModuleDefId::TraitId(t)) => Some(t),
            _ => None,
        })
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

    pub(crate) fn push_res(&mut self, name: Name, res: &Resolution, import: bool) -> bool {
        let mut changed = false;
        let existing = self.visible.entry(name.clone()).or_default();

        if existing.def.types.is_none() && res.def.types.is_some() {
            existing.def.types = res.def.types;
            existing.import = import || res.import;
            changed = true;
        }
        if existing.def.values.is_none() && res.def.values.is_some() {
            existing.def.values = res.def.values;
            existing.import = import || res.import;
            changed = true;
        }
        if existing.def.macros.is_none() && res.def.macros.is_some() {
            existing.def.macros = res.def.macros;
            existing.import = import || res.import;
            changed = true;
        }

        if existing.def.is_none() && res.def.is_none() && !existing.import && res.import {
            existing.import = res.import;
        }
        changed
    }

    pub(crate) fn collect_resolutions(&self) -> Vec<(Name, Resolution)> {
        self.visible.iter().map(|(name, res)| (name.clone(), res.clone())).collect()
    }

    pub(crate) fn collect_legacy_macros(&self) -> FxHashMap<Name, MacroDefId> {
        self.legacy_macros.clone()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Resolution {
    /// None for unresolved
    pub def: PerNs,
    pub(crate) import: bool,
}

impl From<ModuleDefId> for PerNs {
    fn from(def: ModuleDefId) -> PerNs {
        match def {
            ModuleDefId::ModuleId(_) => PerNs::types(def),
            ModuleDefId::FunctionId(_) => PerNs::values(def),
            ModuleDefId::AdtId(adt) => match adt {
                AdtId::StructId(_) | AdtId::UnionId(_) => PerNs::both(def, def),
                AdtId::EnumId(_) => PerNs::types(def),
            },
            ModuleDefId::EnumVariantId(_) => PerNs::both(def, def),
            ModuleDefId::ConstId(_) | ModuleDefId::StaticId(_) => PerNs::values(def),
            ModuleDefId::TraitId(_) => PerNs::types(def),
            ModuleDefId::TypeAliasId(_) => PerNs::types(def),
            ModuleDefId::BuiltinType(_) => PerNs::types(def),
        }
    }
}
