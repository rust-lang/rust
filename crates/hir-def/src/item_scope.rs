//! Describes items defined or visible (ie, imported) in a certain scope.
//! This is shared between modules and blocks.

use std::collections::hash_map::Entry;

use base_db::CrateId;
use hir_expand::{attrs::AttrId, db::ExpandDatabase, name::Name, AstId, MacroCallId};
use itertools::Itertools;
use la_arena::Idx;
use once_cell::sync::Lazy;
use profile::Count;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::{smallvec, SmallVec};
use stdx::format_to;
use syntax::ast;

use crate::{
    db::DefDatabase, per_ns::PerNs, visibility::Visibility, AdtId, BuiltinType, ConstId,
    ExternCrateId, HasModule, ImplId, LocalModuleId, Lookup, MacroId, ModuleDefId, ModuleId,
    TraitId, UseId,
};

#[derive(Debug, Default)]
pub struct PerNsGlobImports {
    types: FxHashSet<(LocalModuleId, Name)>,
    values: FxHashSet<(LocalModuleId, Name)>,
    macros: FxHashSet<(LocalModuleId, Name)>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ImportOrExternCrate {
    Import(ImportId),
    ExternCrate(ExternCrateId),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub(crate) enum ImportType {
    Import(ImportId),
    Glob(UseId),
    ExternCrate(ExternCrateId),
}

impl ImportOrExternCrate {
    pub fn into_import(self) -> Option<ImportId> {
        match self {
            ImportOrExternCrate::Import(it) => Some(it),
            _ => None,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ImportOrDef {
    Import(ImportId),
    ExternCrate(ExternCrateId),
    Def(ModuleDefId),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ImportId {
    pub import: UseId,
    pub idx: Idx<ast::UseTree>,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct ItemScope {
    _c: Count<Self>,

    /// Defs visible in this scope. This includes `declarations`, but also
    /// imports. The imports belong to this module and can be resolved by using them on
    /// the `use_imports_*` fields.
    types: FxHashMap<Name, (ModuleDefId, Visibility, Option<ImportOrExternCrate>)>,
    values: FxHashMap<Name, (ModuleDefId, Visibility, Option<ImportId>)>,
    macros: FxHashMap<Name, (MacroId, Visibility, Option<ImportId>)>,
    unresolved: FxHashSet<Name>,

    /// The defs declared in this scope. Each def has a single scope where it is
    /// declared.
    declarations: Vec<ModuleDefId>,

    impls: Vec<ImplId>,
    unnamed_consts: Vec<ConstId>,
    /// Traits imported via `use Trait as _;`.
    unnamed_trait_imports: FxHashMap<TraitId, (Visibility, Option<ImportId>)>,

    // the resolutions of the imports of this scope
    use_imports_types: FxHashMap<ImportOrExternCrate, ImportOrDef>,
    use_imports_values: FxHashMap<ImportId, ImportOrDef>,
    use_imports_macros: FxHashMap<ImportId, ImportOrDef>,

    use_decls: Vec<UseId>,
    extern_crate_decls: Vec<ExternCrateId>,
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
    legacy_macros: FxHashMap<Name, SmallVec<[MacroId; 1]>>,
    /// The derive macro invocations in this scope.
    attr_macros: FxHashMap<AstId<ast::Item>, MacroCallId>,
    /// The derive macro invocations in this scope, keyed by the owner item over the actual derive attributes
    /// paired with the derive macro invocations for the specific attribute.
    derive_macros: FxHashMap<AstId<ast::Adt>, SmallVec<[DeriveMacroInvocation; 1]>>,
}

#[derive(Debug, PartialEq, Eq)]
struct DeriveMacroInvocation {
    attr_id: AttrId,
    attr_call_id: MacroCallId,
    derive_call_ids: SmallVec<[Option<MacroCallId>; 1]>,
}

pub(crate) static BUILTIN_SCOPE: Lazy<FxHashMap<Name, PerNs>> = Lazy::new(|| {
    BuiltinType::ALL
        .iter()
        .map(|(name, ty)| (name.clone(), PerNs::types((*ty).into(), Visibility::Public, None)))
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
    pub fn entries(&self) -> impl Iterator<Item = (&Name, PerNs)> + '_ {
        // FIXME: shadowing
        self.types
            .keys()
            .chain(self.values.keys())
            .chain(self.macros.keys())
            .chain(self.unresolved.iter())
            .unique()
            .sorted()
            .map(move |name| (name, self.get(name)))
    }

    pub fn imports(&self) -> impl Iterator<Item = ImportId> + '_ {
        self.use_imports_types
            .keys()
            .copied()
            .filter_map(ImportOrExternCrate::into_import)
            .chain(self.use_imports_values.keys().copied())
            .chain(self.use_imports_macros.keys().copied())
            .unique()
            .sorted()
    }

    pub fn fully_resolve_import(&self, db: &dyn DefDatabase, mut import: ImportId) -> PerNs {
        let mut res = PerNs::none();

        let mut def_map;
        let mut scope = self;
        while let Some(&m) = scope.use_imports_macros.get(&import) {
            match m {
                ImportOrDef::Import(i) => {
                    let module_id = i.import.lookup(db).container;
                    def_map = module_id.def_map(db);
                    scope = &def_map[module_id.local_id].scope;
                    import = i;
                }
                ImportOrDef::Def(ModuleDefId::MacroId(def)) => {
                    res.macros = Some((def, Visibility::Public, None));
                    break;
                }
                _ => break,
            }
        }
        let mut scope = self;
        while let Some(&m) = scope.use_imports_types.get(&ImportOrExternCrate::Import(import)) {
            match m {
                ImportOrDef::Import(i) => {
                    let module_id = i.import.lookup(db).container;
                    def_map = module_id.def_map(db);
                    scope = &def_map[module_id.local_id].scope;
                    import = i;
                }
                ImportOrDef::Def(def) => {
                    res.types = Some((def, Visibility::Public, None));
                    break;
                }
                _ => break,
            }
        }
        let mut scope = self;
        while let Some(&m) = scope.use_imports_values.get(&import) {
            match m {
                ImportOrDef::Import(i) => {
                    let module_id = i.import.lookup(db).container;
                    def_map = module_id.def_map(db);
                    scope = &def_map[module_id.local_id].scope;
                    import = i;
                }
                ImportOrDef::Def(def) => {
                    res.values = Some((def, Visibility::Public, None));
                    break;
                }
                _ => break,
            }
        }
        res
    }

    pub fn declarations(&self) -> impl Iterator<Item = ModuleDefId> + '_ {
        self.declarations.iter().copied()
    }

    pub fn extern_crate_decls(
        &self,
    ) -> impl Iterator<Item = ExternCrateId> + ExactSizeIterator + '_ {
        self.extern_crate_decls.iter().copied()
    }

    pub fn use_decls(&self) -> impl Iterator<Item = UseId> + ExactSizeIterator + '_ {
        self.use_decls.iter().copied()
    }

    pub fn impls(&self) -> impl Iterator<Item = ImplId> + ExactSizeIterator + '_ {
        self.impls.iter().copied()
    }

    pub fn values(
        &self,
    ) -> impl Iterator<Item = (ModuleDefId, Visibility)> + ExactSizeIterator + '_ {
        self.values.values().copied().map(|(a, b, _)| (a, b))
    }

    pub(crate) fn types(
        &self,
    ) -> impl Iterator<Item = (ModuleDefId, Visibility)> + ExactSizeIterator + '_ {
        self.types.values().copied().map(|(def, vis, _)| (def, vis))
    }

    pub fn unnamed_consts(&self) -> impl Iterator<Item = ConstId> + '_ {
        self.unnamed_consts.iter().copied()
    }

    /// Iterate over all module scoped macros
    pub(crate) fn macros(&self) -> impl Iterator<Item = (&Name, MacroId)> + '_ {
        self.entries().filter_map(|(name, def)| def.take_macros().map(|macro_| (name, macro_)))
    }

    /// Iterate over all legacy textual scoped macros visible at the end of the module
    pub fn legacy_macros(&self) -> impl Iterator<Item = (&Name, &[MacroId])> + '_ {
        self.legacy_macros.iter().map(|(name, def)| (name, &**def))
    }

    /// Get a name from current module scope, legacy macros are not included
    pub(crate) fn get(&self, name: &Name) -> PerNs {
        PerNs {
            types: self.types.get(name).copied(),
            values: self.values.get(name).copied(),
            macros: self.macros.get(name).copied(),
        }
    }

    pub(crate) fn type_(&self, name: &Name) -> Option<(ModuleDefId, Visibility)> {
        self.types.get(name).copied().map(|(a, b, _)| (a, b))
    }

    /// XXX: this is O(N) rather than O(1), try to not introduce new usages.
    pub(crate) fn name_of(&self, item: ItemInNs) -> Option<(&Name, Visibility)> {
        match item {
            ItemInNs::Macros(def) => self
                .macros
                .iter()
                .find_map(|(name, &(other_def, vis, _))| (other_def == def).then_some((name, vis))),
            ItemInNs::Types(def) => self
                .types
                .iter()
                .find_map(|(name, &(other_def, vis, _))| (other_def == def).then_some((name, vis))),

            ItemInNs::Values(def) => self
                .values
                .iter()
                .find_map(|(name, &(other_def, vis, _))| (other_def == def).then_some((name, vis))),
        }
    }

    pub(crate) fn traits(&self) -> impl Iterator<Item = TraitId> + '_ {
        self.types
            .values()
            .filter_map(|&(def, _, _)| match def {
                ModuleDefId::TraitId(t) => Some(t),
                _ => None,
            })
            .chain(self.unnamed_trait_imports.keys().copied())
    }

    pub(crate) fn resolutions(&self) -> impl Iterator<Item = (Option<Name>, PerNs)> + '_ {
        self.entries().map(|(name, res)| (Some(name.clone()), res)).chain(
            self.unnamed_trait_imports.iter().map(|(tr, (vis, i))| {
                (
                    None,
                    PerNs::types(
                        ModuleDefId::TraitId(*tr),
                        *vis,
                        i.map(ImportOrExternCrate::Import),
                    ),
                )
            }),
        )
    }
}

impl ItemScope {
    pub(crate) fn declare(&mut self, def: ModuleDefId) {
        self.declarations.push(def)
    }

    pub(crate) fn get_legacy_macro(&self, name: &Name) -> Option<&[MacroId]> {
        self.legacy_macros.get(name).map(|it| &**it)
    }

    pub(crate) fn define_impl(&mut self, imp: ImplId) {
        self.impls.push(imp);
    }

    pub(crate) fn define_extern_crate_decl(&mut self, extern_crate: ExternCrateId) {
        self.extern_crate_decls.push(extern_crate);
    }

    pub(crate) fn define_unnamed_const(&mut self, konst: ConstId) {
        self.unnamed_consts.push(konst);
    }

    pub(crate) fn define_legacy_macro(&mut self, name: Name, mac: MacroId) {
        self.legacy_macros.entry(name).or_default().push(mac);
    }

    pub(crate) fn add_attr_macro_invoc(&mut self, item: AstId<ast::Item>, call: MacroCallId) {
        self.attr_macros.insert(item, call);
    }

    pub(crate) fn attr_macro_invocs(
        &self,
    ) -> impl Iterator<Item = (AstId<ast::Item>, MacroCallId)> + '_ {
        self.attr_macros.iter().map(|(k, v)| (*k, *v))
    }

    pub(crate) fn set_derive_macro_invoc(
        &mut self,
        adt: AstId<ast::Adt>,
        call: MacroCallId,
        id: AttrId,
        idx: usize,
    ) {
        if let Some(derives) = self.derive_macros.get_mut(&adt) {
            if let Some(DeriveMacroInvocation { derive_call_ids, .. }) =
                derives.iter_mut().find(|&&mut DeriveMacroInvocation { attr_id, .. }| id == attr_id)
            {
                derive_call_ids[idx] = Some(call);
            }
        }
    }

    /// We are required to set this up front as derive invocation recording happens out of order
    /// due to the fixed pointer iteration loop being able to record some derives later than others
    /// independent of their indices.
    pub(crate) fn init_derive_attribute(
        &mut self,
        adt: AstId<ast::Adt>,
        attr_id: AttrId,
        attr_call_id: MacroCallId,
        len: usize,
    ) {
        self.derive_macros.entry(adt).or_default().push(DeriveMacroInvocation {
            attr_id,
            attr_call_id,
            derive_call_ids: smallvec![None; len],
        });
    }

    pub(crate) fn derive_macro_invocs(
        &self,
    ) -> impl Iterator<
        Item = (
            AstId<ast::Adt>,
            impl Iterator<Item = (AttrId, MacroCallId, &[Option<MacroCallId>])>,
        ),
    > + '_ {
        self.derive_macros.iter().map(|(k, v)| {
            (
                *k,
                v.iter().map(|DeriveMacroInvocation { attr_id, attr_call_id, derive_call_ids }| {
                    (*attr_id, *attr_call_id, &**derive_call_ids)
                }),
            )
        })
    }

    // FIXME: This is only used in collection, we should move the relevant parts of it out of ItemScope
    pub(crate) fn unnamed_trait_vis(&self, tr: TraitId) -> Option<Visibility> {
        self.unnamed_trait_imports.get(&tr).copied().map(|(a, _)| a)
    }

    pub(crate) fn push_unnamed_trait(&mut self, tr: TraitId, vis: Visibility) {
        // FIXME: import
        self.unnamed_trait_imports.insert(tr, (vis, None));
    }

    pub(crate) fn push_res_with_import(
        &mut self,
        glob_imports: &mut PerNsGlobImports,
        lookup: (LocalModuleId, Name),
        def: PerNs,
        import: Option<ImportType>,
    ) -> bool {
        let mut changed = false;

        // FIXME: Document and simplify this

        if let Some(mut fld) = def.types {
            let existing = self.types.entry(lookup.1.clone());
            match existing {
                Entry::Vacant(entry) => {
                    match import {
                        Some(ImportType::Glob(_)) => {
                            glob_imports.types.insert(lookup.clone());
                        }
                        _ => _ = glob_imports.types.remove(&lookup),
                    }
                    let import = match import {
                        Some(ImportType::ExternCrate(extern_crate)) => {
                            Some(ImportOrExternCrate::ExternCrate(extern_crate))
                        }
                        Some(ImportType::Import(import)) => {
                            Some(ImportOrExternCrate::Import(import))
                        }
                        None | Some(ImportType::Glob(_)) => None,
                    };
                    let prev = std::mem::replace(&mut fld.2, import);
                    if let Some(import) = import {
                        self.use_imports_types.insert(
                            import,
                            match prev {
                                Some(ImportOrExternCrate::Import(import)) => {
                                    ImportOrDef::Import(import)
                                }
                                Some(ImportOrExternCrate::ExternCrate(import)) => {
                                    ImportOrDef::ExternCrate(import)
                                }
                                None => ImportOrDef::Def(fld.0),
                            },
                        );
                    }
                    entry.insert(fld);
                    changed = true;
                }
                Entry::Occupied(mut entry) if !matches!(import, Some(ImportType::Glob(..))) => {
                    if glob_imports.types.remove(&lookup) {
                        let import = match import {
                            Some(ImportType::ExternCrate(extern_crate)) => {
                                Some(ImportOrExternCrate::ExternCrate(extern_crate))
                            }
                            Some(ImportType::Import(import)) => {
                                Some(ImportOrExternCrate::Import(import))
                            }
                            None | Some(ImportType::Glob(_)) => None,
                        };
                        let prev = std::mem::replace(&mut fld.2, import);
                        if let Some(import) = import {
                            self.use_imports_types.insert(
                                import,
                                match prev {
                                    Some(ImportOrExternCrate::Import(import)) => {
                                        ImportOrDef::Import(import)
                                    }
                                    Some(ImportOrExternCrate::ExternCrate(import)) => {
                                        ImportOrDef::ExternCrate(import)
                                    }
                                    None => ImportOrDef::Def(fld.0),
                                },
                            );
                        }
                        cov_mark::hit!(import_shadowed);
                        entry.insert(fld);
                        changed = true;
                    }
                }
                _ => {}
            }
        }

        if let Some(mut fld) = def.values {
            let existing = self.values.entry(lookup.1.clone());
            match existing {
                Entry::Vacant(entry) => {
                    match import {
                        Some(ImportType::Glob(_)) => {
                            glob_imports.values.insert(lookup.clone());
                        }
                        _ => _ = glob_imports.values.remove(&lookup),
                    }
                    let import = match import {
                        Some(ImportType::Import(import)) => Some(import),
                        _ => None,
                    };
                    let prev = std::mem::replace(&mut fld.2, import);
                    if let Some(import) = import {
                        self.use_imports_values.insert(
                            import,
                            match prev {
                                Some(import) => ImportOrDef::Import(import),
                                None => ImportOrDef::Def(fld.0),
                            },
                        );
                    }
                    entry.insert(fld);
                    changed = true;
                }
                Entry::Occupied(mut entry) if !matches!(import, Some(ImportType::Glob(..))) => {
                    if glob_imports.values.remove(&lookup) {
                        cov_mark::hit!(import_shadowed);
                        let import = match import {
                            Some(ImportType::Import(import)) => Some(import),
                            _ => None,
                        };
                        let prev = std::mem::replace(&mut fld.2, import);
                        if let Some(import) = import {
                            self.use_imports_values.insert(
                                import,
                                match prev {
                                    Some(import) => ImportOrDef::Import(import),
                                    None => ImportOrDef::Def(fld.0),
                                },
                            );
                        }
                        entry.insert(fld);
                        changed = true;
                    }
                }
                _ => {}
            }
        }

        if let Some(mut fld) = def.macros {
            let existing = self.macros.entry(lookup.1.clone());
            match existing {
                Entry::Vacant(entry) => {
                    match import {
                        Some(ImportType::Glob(_)) => {
                            glob_imports.macros.insert(lookup.clone());
                        }
                        _ => _ = glob_imports.macros.remove(&lookup),
                    }
                    let import = match import {
                        Some(ImportType::Import(import)) => Some(import),
                        _ => None,
                    };
                    let prev = std::mem::replace(&mut fld.2, import);
                    if let Some(import) = import {
                        self.use_imports_macros.insert(
                            import,
                            match prev {
                                Some(import) => ImportOrDef::Import(import),
                                None => ImportOrDef::Def(fld.0.into()),
                            },
                        );
                    }
                    entry.insert(fld);
                    changed = true;
                }
                Entry::Occupied(mut entry) if !matches!(import, Some(ImportType::Glob(..))) => {
                    if glob_imports.macros.remove(&lookup) {
                        cov_mark::hit!(import_shadowed);
                        let import = match import {
                            Some(ImportType::Import(import)) => Some(import),
                            _ => None,
                        };
                        let prev = std::mem::replace(&mut fld.2, import);
                        if let Some(import) = import {
                            self.use_imports_macros.insert(
                                import,
                                match prev {
                                    Some(import) => ImportOrDef::Import(import),
                                    None => ImportOrDef::Def(fld.0.into()),
                                },
                            );
                        }
                        entry.insert(fld);
                        changed = true;
                    }
                }
                _ => {}
            }
        }

        if def.is_none() && self.unresolved.insert(lookup.1) {
            changed = true;
        }

        changed
    }

    /// Marks everything that is not a procedural macro as private to `this_module`.
    pub(crate) fn censor_non_proc_macros(&mut self, this_module: ModuleId) {
        self.types
            .values_mut()
            .map(|(def, vis, _)| (def, vis))
            .chain(self.values.values_mut().map(|(def, vis, _)| (def, vis)))
            .map(|(_, v)| v)
            .chain(self.unnamed_trait_imports.values_mut().map(|(vis, _)| vis))
            .for_each(|vis| *vis = Visibility::Module(this_module));

        for (mac, vis, import) in self.macros.values_mut() {
            if matches!(mac, MacroId::ProcMacroId(_) if import.is_none()) {
                continue;
            }

            *vis = Visibility::Module(this_module);
        }
    }

    pub(crate) fn dump(&self, db: &dyn ExpandDatabase, buf: &mut String) {
        let mut entries: Vec<_> = self.resolutions().collect();
        entries.sort_by_key(|(name, _)| name.clone());

        for (name, def) in entries {
            format_to!(
                buf,
                "{}:",
                name.map_or("_".to_string(), |name| name.display(db).to_string())
            );

            if let Some((.., i)) = def.types {
                buf.push_str(" t");
                match i {
                    Some(ImportOrExternCrate::Import(_)) => buf.push('i'),
                    Some(ImportOrExternCrate::ExternCrate(_)) => buf.push('e'),
                    None => (),
                }
            }
            if let Some((.., i)) = def.values {
                buf.push_str(" v");
                if i.is_some() {
                    buf.push('i');
                }
            }
            if let Some((.., i)) = def.macros {
                buf.push_str(" m");
                if i.is_some() {
                    buf.push('i');
                }
            }
            if def.is_none() {
                buf.push_str(" _");
            }

            buf.push('\n');
        }
    }

    pub(crate) fn shrink_to_fit(&mut self) {
        // Exhaustive match to require handling new fields.
        let Self {
            _c: _,
            types,
            values,
            macros,
            unresolved,
            declarations,
            impls,
            unnamed_consts,
            unnamed_trait_imports,
            legacy_macros,
            attr_macros,
            derive_macros,
            extern_crate_decls,
            use_decls,
            use_imports_values,
            use_imports_types,
            use_imports_macros,
        } = self;
        types.shrink_to_fit();
        values.shrink_to_fit();
        macros.shrink_to_fit();
        use_imports_types.shrink_to_fit();
        use_imports_values.shrink_to_fit();
        use_imports_macros.shrink_to_fit();
        unresolved.shrink_to_fit();
        declarations.shrink_to_fit();
        impls.shrink_to_fit();
        unnamed_consts.shrink_to_fit();
        unnamed_trait_imports.shrink_to_fit();
        legacy_macros.shrink_to_fit();
        attr_macros.shrink_to_fit();
        derive_macros.shrink_to_fit();
        extern_crate_decls.shrink_to_fit();
        use_decls.shrink_to_fit();
    }
}

impl PerNs {
    pub(crate) fn from_def(
        def: ModuleDefId,
        v: Visibility,
        has_constructor: bool,
        import: Option<ImportOrExternCrate>,
    ) -> PerNs {
        match def {
            ModuleDefId::ModuleId(_) => PerNs::types(def, v, import),
            ModuleDefId::FunctionId(_) => {
                PerNs::values(def, v, import.and_then(ImportOrExternCrate::into_import))
            }
            ModuleDefId::AdtId(adt) => match adt {
                AdtId::UnionId(_) => PerNs::types(def, v, import),
                AdtId::EnumId(_) => PerNs::types(def, v, import),
                AdtId::StructId(_) => {
                    if has_constructor {
                        PerNs::both(def, def, v, import)
                    } else {
                        PerNs::types(def, v, import)
                    }
                }
            },
            ModuleDefId::EnumVariantId(_) => PerNs::both(def, def, v, import),
            ModuleDefId::ConstId(_) | ModuleDefId::StaticId(_) => {
                PerNs::values(def, v, import.and_then(ImportOrExternCrate::into_import))
            }
            ModuleDefId::TraitId(_) => PerNs::types(def, v, import),
            ModuleDefId::TraitAliasId(_) => PerNs::types(def, v, import),
            ModuleDefId::TypeAliasId(_) => PerNs::types(def, v, import),
            ModuleDefId::BuiltinType(_) => PerNs::types(def, v, import),
            ModuleDefId::MacroId(mac) => {
                PerNs::macros(mac, v, import.and_then(ImportOrExternCrate::into_import))
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum ItemInNs {
    Types(ModuleDefId),
    Values(ModuleDefId),
    Macros(MacroId),
}

impl ItemInNs {
    pub fn as_module_def_id(self) -> Option<ModuleDefId> {
        match self {
            ItemInNs::Types(id) | ItemInNs::Values(id) => Some(id),
            ItemInNs::Macros(_) => None,
        }
    }

    /// Returns the crate defining this item (or `None` if `self` is built-in).
    pub fn krate(&self, db: &dyn DefDatabase) -> Option<CrateId> {
        match self {
            ItemInNs::Types(id) | ItemInNs::Values(id) => id.module(db).map(|m| m.krate),
            ItemInNs::Macros(id) => Some(id.module(db).krate),
        }
    }

    pub fn module(&self, db: &dyn DefDatabase) -> Option<ModuleId> {
        match self {
            ItemInNs::Types(id) | ItemInNs::Values(id) => id.module(db),
            ItemInNs::Macros(id) => Some(id.module(db)),
        }
    }
}
