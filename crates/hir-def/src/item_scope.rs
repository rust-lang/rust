//! Describes items defined or visible (ie, imported) in a certain scope.
//! This is shared between modules and blocks.

use std::collections::hash_map::Entry;

use base_db::CrateId;
use hir_expand::{attrs::AttrId, db::ExpandDatabase, name::Name, AstId, MacroCallId};
use itertools::Itertools;
use once_cell::sync::Lazy;
use profile::Count;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::{smallvec, SmallVec};
use stdx::format_to;
use syntax::ast;

use crate::{
    db::DefDatabase, per_ns::PerNs, visibility::Visibility, AdtId, BuiltinType, ConstId,
    ExternCrateId, HasModule, ImplId, LocalModuleId, MacroId, ModuleDefId, ModuleId, TraitId,
};

#[derive(Copy, Clone, Debug)]
pub(crate) enum ImportType {
    Glob,
    Named,
}

#[derive(Debug, Default)]
pub struct PerNsGlobImports {
    types: FxHashSet<(LocalModuleId, Name)>,
    values: FxHashSet<(LocalModuleId, Name)>,
    macros: FxHashSet<(LocalModuleId, Name)>,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct ItemScope {
    _c: Count<Self>,

    /// Defs visible in this scope. This includes `declarations`, but also
    /// imports.
    types: FxHashMap<Name, (ModuleDefId, Visibility)>,
    values: FxHashMap<Name, (ModuleDefId, Visibility)>,
    macros: FxHashMap<Name, (MacroId, Visibility)>,
    unresolved: FxHashSet<Name>,

    /// The defs declared in this scope. Each def has a single scope where it is
    /// declared.
    declarations: Vec<ModuleDefId>,

    impls: Vec<ImplId>,
    unnamed_consts: Vec<ConstId>,
    /// Traits imported via `use Trait as _;`.
    unnamed_trait_imports: FxHashMap<TraitId, Visibility>,
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
        .map(|(name, ty)| (name.clone(), PerNs::types((*ty).into(), Visibility::Public)))
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
            .sorted()
            .unique()
            .map(move |name| (name, self.get(name)))
    }

    pub fn declarations(&self) -> impl Iterator<Item = ModuleDefId> + '_ {
        self.declarations.iter().copied()
    }

    pub fn impls(&self) -> impl Iterator<Item = ImplId> + ExactSizeIterator + '_ {
        self.impls.iter().copied()
    }

    pub fn values(
        &self,
    ) -> impl Iterator<Item = (ModuleDefId, Visibility)> + ExactSizeIterator + '_ {
        self.values.values().copied()
    }

    pub fn types(
        &self,
    ) -> impl Iterator<Item = (ModuleDefId, Visibility)> + ExactSizeIterator + '_ {
        self.types.values().copied()
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
        self.types.get(name).copied()
    }

    /// XXX: this is O(N) rather than O(1), try to not introduce new usages.
    pub(crate) fn name_of(&self, item: ItemInNs) -> Option<(&Name, Visibility)> {
        let (def, mut iter) = match item {
            ItemInNs::Macros(def) => {
                return self.macros.iter().find_map(|(name, &(other_def, vis))| {
                    (other_def == def).then_some((name, vis))
                });
            }
            ItemInNs::Types(def) => (def, self.types.iter()),
            ItemInNs::Values(def) => (def, self.values.iter()),
        };
        iter.find_map(|(name, &(other_def, vis))| (other_def == def).then_some((name, vis)))
    }

    pub(crate) fn traits(&self) -> impl Iterator<Item = TraitId> + '_ {
        self.types
            .values()
            .filter_map(|&(def, _)| match def {
                ModuleDefId::TraitId(t) => Some(t),
                _ => None,
            })
            .chain(self.unnamed_trait_imports.keys().copied())
    }

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

    pub(crate) fn unnamed_trait_vis(&self, tr: TraitId) -> Option<Visibility> {
        self.unnamed_trait_imports.get(&tr).copied()
    }

    pub(crate) fn push_unnamed_trait(&mut self, tr: TraitId, vis: Visibility) {
        self.unnamed_trait_imports.insert(tr, vis);
    }

    pub(crate) fn push_res_with_import(
        &mut self,
        glob_imports: &mut PerNsGlobImports,
        lookup: (LocalModuleId, Name),
        def: PerNs,
        def_import_type: ImportType,
    ) -> bool {
        let mut changed = false;

        macro_rules! check_changed {
            (
                $changed:ident,
                ( $this:ident / $def:ident ) . $field:ident,
                $glob_imports:ident [ $lookup:ident ],
                $def_import_type:ident
            ) => {{
                if let Some(fld) = $def.$field {
                    let existing = $this.$field.entry($lookup.1.clone());
                    match existing {
                        Entry::Vacant(entry) => {
                            match $def_import_type {
                                ImportType::Glob => {
                                    $glob_imports.$field.insert($lookup.clone());
                                }
                                ImportType::Named => {
                                    $glob_imports.$field.remove(&$lookup);
                                }
                            }

                            entry.insert(fld);
                            $changed = true;
                        }
                        Entry::Occupied(mut entry)
                            if matches!($def_import_type, ImportType::Named) =>
                        {
                            if $glob_imports.$field.remove(&$lookup) {
                                cov_mark::hit!(import_shadowed);
                                entry.insert(fld);
                                $changed = true;
                            }
                        }
                        _ => {}
                    }
                }
            }};
        }

        check_changed!(changed, (self / def).types, glob_imports[lookup], def_import_type);
        check_changed!(changed, (self / def).values, glob_imports[lookup], def_import_type);
        check_changed!(changed, (self / def).macros, glob_imports[lookup], def_import_type);

        if def.is_none() && self.unresolved.insert(lookup.1) {
            changed = true;
        }

        changed
    }

    pub(crate) fn resolutions(&self) -> impl Iterator<Item = (Option<Name>, PerNs)> + '_ {
        self.entries().map(|(name, res)| (Some(name.clone()), res)).chain(
            self.unnamed_trait_imports
                .iter()
                .map(|(tr, vis)| (None, PerNs::types(ModuleDefId::TraitId(*tr), *vis))),
        )
    }

    /// Marks everything that is not a procedural macro as private to `this_module`.
    pub(crate) fn censor_non_proc_macros(&mut self, this_module: ModuleId) {
        self.types
            .values_mut()
            .chain(self.values.values_mut())
            .map(|(_, v)| v)
            .chain(self.unnamed_trait_imports.values_mut())
            .for_each(|vis| *vis = Visibility::Module(this_module));

        for (mac, vis) in self.macros.values_mut() {
            if let MacroId::ProcMacroId(_) = mac {
                // FIXME: Technically this is insufficient since reexports of proc macros are also
                // forbidden. Practically nobody does that.
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

            if def.types.is_some() {
                buf.push_str(" t");
            }
            if def.values.is_some() {
                buf.push_str(" v");
            }
            if def.macros.is_some() {
                buf.push_str(" m");
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
        } = self;
        types.shrink_to_fit();
        values.shrink_to_fit();
        macros.shrink_to_fit();
        unresolved.shrink_to_fit();
        declarations.shrink_to_fit();
        impls.shrink_to_fit();
        unnamed_consts.shrink_to_fit();
        unnamed_trait_imports.shrink_to_fit();
        legacy_macros.shrink_to_fit();
        attr_macros.shrink_to_fit();
        derive_macros.shrink_to_fit();
        extern_crate_decls.shrink_to_fit();
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
            ModuleDefId::TraitAliasId(_) => PerNs::types(def, v),
            ModuleDefId::TypeAliasId(_) => PerNs::types(def, v),
            ModuleDefId::BuiltinType(_) => PerNs::types(def, v),
            ModuleDefId::MacroId(mac) => PerNs::macros(mac, v),
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
