//! Describes items defined or visible (ie, imported) in a certain scope.
//! This is shared between modules and blocks.

use std::sync::LazyLock;

use base_db::Crate;
use hir_expand::{AstId, MacroCallId, attrs::AttrId, db::ExpandDatabase, name::Name};
use indexmap::map::Entry;
use itertools::Itertools;
use la_arena::Idx;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::{SmallVec, smallvec};
use span::Edition;
use stdx::format_to;
use syntax::ast;
use thin_vec::ThinVec;

use crate::{
    AdtId, BuiltinType, ConstId, ExternBlockId, ExternCrateId, FxIndexMap, HasModule, ImplId,
    LocalModuleId, Lookup, MacroId, ModuleDefId, ModuleId, TraitId, UseId,
    db::DefDatabase,
    per_ns::{Item, MacrosItem, PerNs, TypesItem, ValuesItem},
    visibility::Visibility,
};

#[derive(Debug, Default)]
pub struct PerNsGlobImports {
    types: FxHashSet<(LocalModuleId, Name)>,
    values: FxHashSet<(LocalModuleId, Name)>,
    macros: FxHashSet<(LocalModuleId, Name)>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ImportOrExternCrate {
    Glob(GlobId),
    Import(ImportId),
    ExternCrate(ExternCrateId),
}

impl From<ImportOrGlob> for ImportOrExternCrate {
    fn from(value: ImportOrGlob) -> Self {
        match value {
            ImportOrGlob::Glob(it) => ImportOrExternCrate::Glob(it),
            ImportOrGlob::Import(it) => ImportOrExternCrate::Import(it),
        }
    }
}

impl ImportOrExternCrate {
    pub fn import_or_glob(self) -> Option<ImportOrGlob> {
        match self {
            ImportOrExternCrate::Import(it) => Some(ImportOrGlob::Import(it)),
            ImportOrExternCrate::Glob(it) => Some(ImportOrGlob::Glob(it)),
            _ => None,
        }
    }

    pub fn import(self) -> Option<ImportId> {
        match self {
            ImportOrExternCrate::Import(it) => Some(it),
            _ => None,
        }
    }

    pub fn glob(self) -> Option<GlobId> {
        match self {
            ImportOrExternCrate::Glob(id) => Some(id),
            _ => None,
        }
    }

    pub fn use_(self) -> Option<UseId> {
        match self {
            ImportOrExternCrate::Glob(id) => Some(id.use_),
            ImportOrExternCrate::Import(id) => Some(id.use_),
            _ => None,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ImportOrGlob {
    Glob(GlobId),
    Import(ImportId),
}

impl ImportOrGlob {
    pub fn into_import(self) -> Option<ImportId> {
        match self {
            ImportOrGlob::Import(it) => Some(it),
            _ => None,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ImportOrDef {
    Import(ImportId),
    Glob(GlobId),
    ExternCrate(ExternCrateId),
    Def(ModuleDefId),
}

impl From<ImportOrExternCrate> for ImportOrDef {
    fn from(value: ImportOrExternCrate) -> Self {
        match value {
            ImportOrExternCrate::Import(it) => ImportOrDef::Import(it),
            ImportOrExternCrate::Glob(it) => ImportOrDef::Glob(it),
            ImportOrExternCrate::ExternCrate(it) => ImportOrDef::ExternCrate(it),
        }
    }
}

impl From<ImportOrGlob> for ImportOrDef {
    fn from(value: ImportOrGlob) -> Self {
        match value {
            ImportOrGlob::Import(it) => ImportOrDef::Import(it),
            ImportOrGlob::Glob(it) => ImportOrDef::Glob(it),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ImportId {
    pub use_: UseId,
    pub idx: Idx<ast::UseTree>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct GlobId {
    pub use_: UseId,
    pub idx: Idx<ast::UseTree>,
}

impl PerNsGlobImports {
    pub(crate) fn contains_type(&self, module_id: LocalModuleId, name: Name) -> bool {
        self.types.contains(&(module_id, name))
    }
    pub(crate) fn contains_value(&self, module_id: LocalModuleId, name: Name) -> bool {
        self.values.contains(&(module_id, name))
    }
    pub(crate) fn contains_macro(&self, module_id: LocalModuleId, name: Name) -> bool {
        self.macros.contains(&(module_id, name))
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct ItemScope {
    /// Defs visible in this scope. This includes `declarations`, but also
    /// imports. The imports belong to this module and can be resolved by using them on
    /// the `use_imports_*` fields.
    types: FxIndexMap<Name, TypesItem>,
    values: FxIndexMap<Name, ValuesItem>,
    macros: FxIndexMap<Name, MacrosItem>,
    unresolved: FxHashSet<Name>,

    /// The defs declared in this scope. Each def has a single scope where it is
    /// declared.
    declarations: ThinVec<ModuleDefId>,

    impls: ThinVec<ImplId>,
    extern_blocks: ThinVec<ExternBlockId>,
    unnamed_consts: ThinVec<ConstId>,
    /// Traits imported via `use Trait as _;`.
    unnamed_trait_imports: ThinVec<(TraitId, Item<()>)>,

    // the resolutions of the imports of this scope
    use_imports_types: FxHashMap<ImportOrExternCrate, ImportOrDef>,
    use_imports_values: FxHashMap<ImportOrGlob, ImportOrDef>,
    use_imports_macros: FxHashMap<ImportOrExternCrate, ImportOrDef>,

    use_decls: ThinVec<UseId>,
    extern_crate_decls: ThinVec<ExternCrateId>,
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
    legacy_macros: FxHashMap<Name, SmallVec<[MacroId; 2]>>,
    /// The attribute macro invocations in this scope.
    attr_macros: FxHashMap<AstId<ast::Item>, MacroCallId>,
    /// The macro invocations in this scope.
    macro_invocations: FxHashMap<AstId<ast::MacroCall>, MacroCallId>,
    /// The derive macro invocations in this scope, keyed by the owner item over the actual derive attributes
    /// paired with the derive macro invocations for the specific attribute.
    derive_macros: FxHashMap<AstId<ast::Adt>, SmallVec<[DeriveMacroInvocation; 1]>>,
}

#[derive(Debug, PartialEq, Eq)]
struct DeriveMacroInvocation {
    attr_id: AttrId,
    /// The `#[derive]` call
    attr_call_id: MacroCallId,
    derive_call_ids: SmallVec<[Option<MacroCallId>; 4]>,
}

pub(crate) static BUILTIN_SCOPE: LazyLock<FxIndexMap<Name, PerNs>> = LazyLock::new(|| {
    BuiltinType::all_builtin_types()
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
            .sorted()
            .dedup()
            .map(move |name| (name, self.get(name)))
    }

    pub fn values(&self) -> impl Iterator<Item = (&Name, Item<ModuleDefId, ImportOrGlob>)> + '_ {
        self.values.iter().map(|(n, &i)| (n, i))
    }

    pub fn types(
        &self,
    ) -> impl Iterator<Item = (&Name, Item<ModuleDefId, ImportOrExternCrate>)> + '_ {
        self.types.iter().map(|(n, &i)| (n, i))
    }

    pub fn macros(&self) -> impl Iterator<Item = (&Name, Item<MacroId, ImportOrExternCrate>)> + '_ {
        self.macros.iter().map(|(n, &i)| (n, i))
    }

    pub fn imports(&self) -> impl Iterator<Item = ImportId> + '_ {
        self.use_imports_types
            .keys()
            .copied()
            .chain(self.use_imports_macros.keys().copied())
            .filter_map(ImportOrExternCrate::import_or_glob)
            .chain(self.use_imports_values.keys().copied())
            .filter_map(ImportOrGlob::into_import)
            .sorted()
            .dedup()
    }

    pub fn fully_resolve_import(&self, db: &dyn DefDatabase, mut import: ImportId) -> PerNs {
        let mut res = PerNs::none();

        let mut def_map;
        let mut scope = self;
        while let Some(&m) = scope.use_imports_macros.get(&ImportOrExternCrate::Import(import)) {
            match m {
                ImportOrDef::Import(i) => {
                    let module_id = i.use_.lookup(db).container;
                    def_map = module_id.def_map(db);
                    scope = &def_map[module_id.local_id].scope;
                    import = i;
                }
                ImportOrDef::Def(ModuleDefId::MacroId(def)) => {
                    res.macros = Some(Item { def, vis: Visibility::Public, import: None });
                    break;
                }
                _ => break,
            }
        }
        let mut scope = self;
        while let Some(&m) = scope.use_imports_types.get(&ImportOrExternCrate::Import(import)) {
            match m {
                ImportOrDef::Import(i) => {
                    let module_id = i.use_.lookup(db).container;
                    def_map = module_id.def_map(db);
                    scope = &def_map[module_id.local_id].scope;
                    import = i;
                }
                ImportOrDef::Def(def) => {
                    res.types = Some(Item { def, vis: Visibility::Public, import: None });
                    break;
                }
                _ => break,
            }
        }
        let mut scope = self;
        while let Some(&m) = scope.use_imports_values.get(&ImportOrGlob::Import(import)) {
            match m {
                ImportOrDef::Import(i) => {
                    let module_id = i.use_.lookup(db).container;
                    def_map = module_id.def_map(db);
                    scope = &def_map[module_id.local_id].scope;
                    import = i;
                }
                ImportOrDef::Def(def) => {
                    res.values = Some(Item { def, vis: Visibility::Public, import: None });
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

    pub fn extern_crate_decls(&self) -> impl ExactSizeIterator<Item = ExternCrateId> + '_ {
        self.extern_crate_decls.iter().copied()
    }

    pub fn extern_blocks(&self) -> impl Iterator<Item = ExternBlockId> + '_ {
        self.extern_blocks.iter().copied()
    }

    pub fn use_decls(&self) -> impl ExactSizeIterator<Item = UseId> + '_ {
        self.use_decls.iter().copied()
    }

    pub fn impls(&self) -> impl ExactSizeIterator<Item = ImplId> + '_ {
        self.impls.iter().copied()
    }

    pub fn all_macro_calls(&self) -> impl Iterator<Item = MacroCallId> + '_ {
        self.macro_invocations.values().copied().chain(self.attr_macros.values().copied()).chain(
            self.derive_macros.values().flat_map(|it| {
                it.iter().flat_map(|it| it.derive_call_ids.iter().copied().flatten())
            }),
        )
    }

    pub(crate) fn modules_in_scope(&self) -> impl Iterator<Item = (ModuleId, Visibility)> + '_ {
        self.types.values().filter_map(|ns| match ns.def {
            ModuleDefId::ModuleId(module) => Some((module, ns.vis)),
            _ => None,
        })
    }

    pub fn unnamed_consts(&self) -> impl Iterator<Item = ConstId> + '_ {
        self.unnamed_consts.iter().copied()
    }

    /// Iterate over all legacy textual scoped macros visible at the end of the module
    pub fn legacy_macros(&self) -> impl Iterator<Item = (&Name, &[MacroId])> + '_ {
        self.legacy_macros.iter().map(|(name, def)| (name, &**def))
    }

    /// Get a name from current module scope, legacy macros are not included
    pub fn get(&self, name: &Name) -> PerNs {
        PerNs {
            types: self.types.get(name).copied(),
            values: self.values.get(name).copied(),
            macros: self.macros.get(name).copied(),
        }
    }

    pub(crate) fn type_(&self, name: &Name) -> Option<(ModuleDefId, Visibility)> {
        self.types.get(name).map(|item| (item.def, item.vis))
    }

    /// XXX: this is O(N) rather than O(1), try to not introduce new usages.
    pub(crate) fn name_of(&self, item: ItemInNs) -> Option<(&Name, Visibility, /*declared*/ bool)> {
        match item {
            ItemInNs::Macros(def) => self.macros.iter().find_map(|(name, other_def)| {
                (other_def.def == def).then_some((name, other_def.vis, other_def.import.is_none()))
            }),
            ItemInNs::Types(def) => self.types.iter().find_map(|(name, other_def)| {
                (other_def.def == def).then_some((name, other_def.vis, other_def.import.is_none()))
            }),
            ItemInNs::Values(def) => self.values.iter().find_map(|(name, other_def)| {
                (other_def.def == def).then_some((name, other_def.vis, other_def.import.is_none()))
            }),
        }
    }

    /// XXX: this is O(N) rather than O(1), try to not introduce new usages.
    pub(crate) fn names_of<T>(
        &self,
        item: ItemInNs,
        mut cb: impl FnMut(&Name, Visibility, /*declared*/ bool) -> Option<T>,
    ) -> Option<T> {
        match item {
            ItemInNs::Macros(def) => self
                .macros
                .iter()
                .filter_map(|(name, other_def)| {
                    (other_def.def == def).then_some((
                        name,
                        other_def.vis,
                        other_def.import.is_none(),
                    ))
                })
                .find_map(|(a, b, c)| cb(a, b, c)),
            ItemInNs::Types(def) => self
                .types
                .iter()
                .filter_map(|(name, other_def)| {
                    (other_def.def == def).then_some((
                        name,
                        other_def.vis,
                        other_def.import.is_none(),
                    ))
                })
                .find_map(|(a, b, c)| cb(a, b, c)),
            ItemInNs::Values(def) => self
                .values
                .iter()
                .filter_map(|(name, other_def)| {
                    (other_def.def == def).then_some((
                        name,
                        other_def.vis,
                        other_def.import.is_none(),
                    ))
                })
                .find_map(|(a, b, c)| cb(a, b, c)),
        }
    }

    pub(crate) fn traits(&self) -> impl Iterator<Item = TraitId> + '_ {
        self.types
            .values()
            .filter_map(|def| match def.def {
                ModuleDefId::TraitId(t) => Some(t),
                _ => None,
            })
            .chain(self.unnamed_trait_imports.iter().map(|&(t, _)| t))
    }

    pub(crate) fn resolutions(&self) -> impl Iterator<Item = (Option<Name>, PerNs)> + '_ {
        self.entries().map(|(name, res)| (Some(name.clone()), res)).chain(
            self.unnamed_trait_imports.iter().map(|(tr, trait_)| {
                (
                    None,
                    PerNs::types(
                        ModuleDefId::TraitId(*tr),
                        trait_.vis,
                        trait_.import.map(ImportOrExternCrate::Import),
                    ),
                )
            }),
        )
    }

    pub fn macro_invoc(&self, call: AstId<ast::MacroCall>) -> Option<MacroCallId> {
        self.macro_invocations.get(&call).copied()
    }

    pub fn iter_macro_invoc(&self) -> impl Iterator<Item = (&AstId<ast::MacroCall>, &MacroCallId)> {
        self.macro_invocations.iter()
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

    pub(crate) fn define_extern_block(&mut self, extern_block: ExternBlockId) {
        self.extern_blocks.push(extern_block);
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

    pub(crate) fn add_macro_invoc(&mut self, call: AstId<ast::MacroCall>, call_id: MacroCallId) {
        self.macro_invocations.insert(call, call_id);
    }

    pub fn attr_macro_invocs(&self) -> impl Iterator<Item = (AstId<ast::Item>, MacroCallId)> + '_ {
        self.attr_macros.iter().map(|(k, v)| (*k, *v))
    }

    pub(crate) fn set_derive_macro_invoc(
        &mut self,
        adt: AstId<ast::Adt>,
        call: MacroCallId,
        id: AttrId,
        idx: usize,
    ) {
        if let Some(derives) = self.derive_macros.get_mut(&adt)
            && let Some(DeriveMacroInvocation { derive_call_ids, .. }) =
                derives.iter_mut().find(|&&mut DeriveMacroInvocation { attr_id, .. }| id == attr_id)
        {
            derive_call_ids[idx] = Some(call);
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

    pub fn derive_macro_invocs(
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

    pub fn derive_macro_invoc(
        &self,
        ast_id: AstId<ast::Adt>,
        attr_id: AttrId,
    ) -> Option<MacroCallId> {
        Some(self.derive_macros.get(&ast_id)?.iter().find(|it| it.attr_id == attr_id)?.attr_call_id)
    }

    // FIXME: This is only used in collection, we should move the relevant parts of it out of ItemScope
    pub(crate) fn unnamed_trait_vis(&self, tr: TraitId) -> Option<Visibility> {
        self.unnamed_trait_imports.iter().find(|&&(t, _)| t == tr).map(|(_, trait_)| trait_.vis)
    }

    pub(crate) fn push_unnamed_trait(
        &mut self,
        tr: TraitId,
        vis: Visibility,
        import: Option<ImportId>,
    ) {
        self.unnamed_trait_imports.push((tr, Item { def: (), vis, import }));
    }

    pub(crate) fn push_res_with_import(
        &mut self,
        glob_imports: &mut PerNsGlobImports,
        lookup: (LocalModuleId, Name),
        def: PerNs,
        import: Option<ImportOrExternCrate>,
    ) -> bool {
        let mut changed = false;

        // FIXME: Document and simplify this

        if let Some(mut fld) = def.types {
            let existing = self.types.entry(lookup.1.clone());
            match existing {
                Entry::Vacant(entry) => {
                    match import {
                        Some(ImportOrExternCrate::Glob(_)) => {
                            glob_imports.types.insert(lookup.clone());
                        }
                        _ => _ = glob_imports.types.remove(&lookup),
                    }
                    let prev = std::mem::replace(&mut fld.import, import);
                    if let Some(import) = import {
                        self.use_imports_types
                            .insert(import, prev.map_or(ImportOrDef::Def(fld.def), Into::into));
                    }
                    entry.insert(fld);
                    changed = true;
                }
                Entry::Occupied(mut entry) => {
                    match import {
                        Some(ImportOrExternCrate::Glob(..)) => {
                            // Multiple globs may import the same item and they may
                            // override visibility from previously resolved globs. This is
                            // currently handled by `DefCollector`, because we need to
                            // compute the max visibility for items and we need `DefMap`
                            // for that.
                        }
                        _ => {
                            if glob_imports.types.remove(&lookup) {
                                let prev = std::mem::replace(&mut fld.import, import);
                                if let Some(import) = import {
                                    self.use_imports_types.insert(
                                        import,
                                        prev.map_or(ImportOrDef::Def(fld.def), Into::into),
                                    );
                                }
                                cov_mark::hit!(import_shadowed);
                                entry.insert(fld);
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        if let Some(mut fld) = def.values {
            let existing = self.values.entry(lookup.1.clone());
            match existing {
                Entry::Vacant(entry) => {
                    match import {
                        Some(ImportOrExternCrate::Glob(_)) => {
                            glob_imports.values.insert(lookup.clone());
                        }
                        _ => _ = glob_imports.values.remove(&lookup),
                    }
                    let import = import.and_then(ImportOrExternCrate::import_or_glob);
                    let prev = std::mem::replace(&mut fld.import, import);
                    if let Some(import) = import {
                        self.use_imports_values
                            .insert(import, prev.map_or(ImportOrDef::Def(fld.def), Into::into));
                    }
                    entry.insert(fld);
                    changed = true;
                }
                Entry::Occupied(mut entry)
                    if !matches!(import, Some(ImportOrExternCrate::Glob(..))) =>
                {
                    if glob_imports.values.remove(&lookup) {
                        cov_mark::hit!(import_shadowed);

                        let import = import.and_then(ImportOrExternCrate::import_or_glob);
                        let prev = std::mem::replace(&mut fld.import, import);
                        if let Some(import) = import {
                            self.use_imports_values
                                .insert(import, prev.map_or(ImportOrDef::Def(fld.def), Into::into));
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
                        Some(ImportOrExternCrate::Glob(_)) => {
                            glob_imports.macros.insert(lookup.clone());
                        }
                        _ => _ = glob_imports.macros.remove(&lookup),
                    }
                    let prev = std::mem::replace(&mut fld.import, import);
                    if let Some(import) = import {
                        self.use_imports_macros.insert(
                            import,
                            prev.map_or_else(|| ImportOrDef::Def(fld.def.into()), Into::into),
                        );
                    }
                    entry.insert(fld);
                    changed = true;
                }
                Entry::Occupied(mut entry)
                    if !matches!(import, Some(ImportOrExternCrate::Glob(..))) =>
                {
                    if glob_imports.macros.remove(&lookup) {
                        cov_mark::hit!(import_shadowed);
                        let prev = std::mem::replace(&mut fld.import, import);
                        if let Some(import) = import {
                            self.use_imports_macros.insert(
                                import,
                                prev.map_or_else(|| ImportOrDef::Def(fld.def.into()), Into::into),
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
    pub(crate) fn censor_non_proc_macros(&mut self, krate: Crate) {
        self.types
            .values_mut()
            .map(|def| &mut def.vis)
            .chain(self.values.values_mut().map(|def| &mut def.vis))
            .chain(self.unnamed_trait_imports.iter_mut().map(|(_, def)| &mut def.vis))
            .for_each(|vis| *vis = Visibility::PubCrate(krate));

        for mac in self.macros.values_mut() {
            if matches!(mac.def, MacroId::ProcMacroId(_) if mac.import.is_none()) {
                continue;
            }
            mac.vis = Visibility::PubCrate(krate)
        }
    }

    pub(crate) fn dump(&self, db: &dyn ExpandDatabase, buf: &mut String) {
        let mut entries: Vec<_> = self.resolutions().collect();
        entries.sort_by_key(|(name, _)| name.clone());

        for (name, def) in entries {
            format_to!(
                buf,
                "{}:",
                name.map_or("_".to_owned(), |name| name.display(db, Edition::LATEST).to_string())
            );

            if let Some(Item { import, .. }) = def.types {
                buf.push_str(" t");
                match import {
                    Some(ImportOrExternCrate::Import(_)) => buf.push('i'),
                    Some(ImportOrExternCrate::Glob(_)) => buf.push('g'),
                    Some(ImportOrExternCrate::ExternCrate(_)) => buf.push('e'),
                    None => (),
                }
            }
            if let Some(Item { import, .. }) = def.values {
                buf.push_str(" v");
                match import {
                    Some(ImportOrGlob::Import(_)) => buf.push('i'),
                    Some(ImportOrGlob::Glob(_)) => buf.push('g'),
                    None => (),
                }
            }
            if let Some(Item { import, .. }) = def.macros {
                buf.push_str(" m");
                match import {
                    Some(ImportOrExternCrate::Import(_)) => buf.push('i'),
                    Some(ImportOrExternCrate::Glob(_)) => buf.push('g'),
                    Some(ImportOrExternCrate::ExternCrate(_)) => buf.push('e'),
                    None => (),
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
            macro_invocations,
            extern_blocks,
        } = self;
        extern_blocks.shrink_to_fit();
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
        macro_invocations.shrink_to_fit();
    }
}

// These methods are a temporary measure only meant to be used by `DefCollector::push_res_and_update_glob_vis()`.
impl ItemScope {
    pub(crate) fn update_visibility_types(&mut self, name: &Name, vis: Visibility) {
        let res =
            self.types.get_mut(name).expect("tried to update visibility of non-existent type");
        res.vis = vis;
    }

    pub(crate) fn update_visibility_values(&mut self, name: &Name, vis: Visibility) {
        let res =
            self.values.get_mut(name).expect("tried to update visibility of non-existent value");
        res.vis = vis;
    }

    pub(crate) fn update_visibility_macros(&mut self, name: &Name, vis: Visibility) {
        let res =
            self.macros.get_mut(name).expect("tried to update visibility of non-existent macro");
        res.vis = vis;
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
                PerNs::values(def, v, import.and_then(ImportOrExternCrate::import_or_glob))
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
                PerNs::values(def, v, import.and_then(ImportOrExternCrate::import_or_glob))
            }
            ModuleDefId::TraitId(_) => PerNs::types(def, v, import),
            ModuleDefId::TraitAliasId(_) => PerNs::types(def, v, import),
            ModuleDefId::TypeAliasId(_) => PerNs::types(def, v, import),
            ModuleDefId::BuiltinType(_) => PerNs::types(def, v, import),
            ModuleDefId::MacroId(mac) => PerNs::macros(mac, v, import),
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
    pub fn krate(&self, db: &dyn DefDatabase) -> Option<Crate> {
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
