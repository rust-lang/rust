//! Look up accessible paths for items.
use either::Either;
use hir::{
    AsAssocItem, AssocItem, AssocItemContainer, Crate, ItemInNs, MacroDef, ModPath, Module,
    ModuleDef, PathResolution, PrefixKind, ScopeDef, Semantics, SemanticsScope, Type,
};
use rustc_hash::FxHashSet;
use syntax::{ast, AstNode};

use crate::{
    imports_locator::{self, AssocItemSearch, DEFAULT_QUERY_SEARCH_LIMIT},
    RootDatabase,
};

use super::item_name;

#[derive(Debug)]
pub enum ImportCandidate {
    // A path, qualified (`std::collections::HashMap`) or not (`HashMap`).
    Path(PathImportCandidate),
    /// A trait associated function (with no self parameter) or associated constant.
    /// For 'test_mod::TestEnum::test_function', `ty` is the `test_mod::TestEnum` expression type
    /// and `name` is the `test_function`
    TraitAssocItem(TraitImportCandidate),
    /// A trait method with self parameter.
    /// For 'test_enum.test_method()', `ty` is the `test_enum` expression type
    /// and `name` is the `test_method`
    TraitMethod(TraitImportCandidate),
}

#[derive(Debug)]
pub struct TraitImportCandidate {
    pub receiver_ty: Type,
    pub name: NameToImport,
}

#[derive(Debug)]
pub struct PathImportCandidate {
    pub qualifier: Qualifier,
    pub name: NameToImport,
}

#[derive(Debug)]
pub enum Qualifier {
    Absent,
    FirstSegmentUnresolved(ast::NameRef, ModPath),
}

#[derive(Debug)]
pub enum NameToImport {
    Exact(String),
    Fuzzy(String),
}

impl NameToImport {
    pub fn text(&self) -> &str {
        match self {
            NameToImport::Exact(text) => text.as_str(),
            NameToImport::Fuzzy(text) => text.as_str(),
        }
    }
}

#[derive(Debug)]
pub struct ImportAssets<'a> {
    import_candidate: ImportCandidate,
    module_with_candidate: Module,
    scope: SemanticsScope<'a>,
}

impl<'a> ImportAssets<'a> {
    pub fn for_method_call(
        method_call: &ast::MethodCallExpr,
        sema: &'a Semantics<RootDatabase>,
    ) -> Option<Self> {
        let scope = sema.scope(method_call.syntax());
        Some(Self {
            import_candidate: ImportCandidate::for_method_call(sema, method_call)?,
            module_with_candidate: scope.module()?,
            scope,
        })
    }

    pub fn for_exact_path(
        fully_qualified_path: &ast::Path,
        sema: &'a Semantics<RootDatabase>,
    ) -> Option<Self> {
        let syntax_under_caret = fully_qualified_path.syntax();
        if syntax_under_caret.ancestors().find_map(ast::Use::cast).is_some() {
            return None;
        }
        let scope = sema.scope(syntax_under_caret);
        Some(Self {
            import_candidate: ImportCandidate::for_regular_path(sema, fully_qualified_path)?,
            module_with_candidate: scope.module()?,
            scope,
        })
    }

    pub fn for_fuzzy_path(
        module_with_candidate: Module,
        qualifier: Option<ast::Path>,
        fuzzy_name: String,
        sema: &Semantics<RootDatabase>,
        scope: SemanticsScope<'a>,
    ) -> Option<Self> {
        Some(Self {
            import_candidate: ImportCandidate::for_fuzzy_path(qualifier, fuzzy_name, sema)?,
            module_with_candidate,
            scope,
        })
    }

    pub fn for_fuzzy_method_call(
        module_with_method_call: Module,
        receiver_ty: Type,
        fuzzy_method_name: String,
        scope: SemanticsScope<'a>,
    ) -> Option<Self> {
        Some(Self {
            import_candidate: ImportCandidate::TraitMethod(TraitImportCandidate {
                receiver_ty,
                name: NameToImport::Fuzzy(fuzzy_method_name),
            }),
            module_with_candidate: module_with_method_call,
            scope,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LocatedImport {
    // TODO kb extract both into a separate struct + add another field:  `assoc_item_name: Optional<String|Name>`
    import_path: ModPath,
    item_to_import: ItemInNs,
    data_to_display: Option<(ModPath, ItemInNs)>,
}

impl LocatedImport {
    pub fn new(
        import_path: ModPath,
        item_to_import: ItemInNs,
        data_to_display: Option<(ModPath, ItemInNs)>,
    ) -> Self {
        Self { import_path, item_to_import, data_to_display }
    }

    pub fn display_path(&self) -> &ModPath {
        self.data_to_display.as_ref().map(|(mod_pathh, _)| mod_pathh).unwrap_or(&self.import_path)
    }

    pub fn import_path(&self) -> &ModPath {
        &self.import_path
    }

    pub fn item_to_display(&self) -> ItemInNs {
        self.data_to_display.as_ref().map(|&(_, item)| item).unwrap_or(self.item_to_import)
    }

    pub fn item_to_import(&self) -> ItemInNs {
        self.item_to_import
    }
}

impl<'a> ImportAssets<'a> {
    pub fn import_candidate(&self) -> &ImportCandidate {
        &self.import_candidate
    }

    fn name_to_import(&self) -> &NameToImport {
        match &self.import_candidate {
            ImportCandidate::Path(candidate) => &candidate.name,
            ImportCandidate::TraitAssocItem(candidate)
            | ImportCandidate::TraitMethod(candidate) => &candidate.name,
        }
    }

    pub fn search_for_imports(
        &self,
        sema: &Semantics<RootDatabase>,
        prefix_kind: PrefixKind,
    ) -> Vec<LocatedImport> {
        let _p = profile::span("import_assets::search_for_imports");
        self.search_for(sema, Some(prefix_kind))
    }

    /// This may return non-absolute paths if a part of the returned path is already imported into scope.
    pub fn search_for_relative_paths(&self, sema: &Semantics<RootDatabase>) -> Vec<LocatedImport> {
        let _p = profile::span("import_assets::search_for_relative_paths");
        self.search_for(sema, None)
    }

    fn search_for(
        &self,
        sema: &Semantics<RootDatabase>,
        prefixed: Option<PrefixKind>,
    ) -> Vec<LocatedImport> {
        let current_crate = self.module_with_candidate.krate();
        let scope_definitions = self.scope_definitions();

        let defs_for_candidate_name = match self.name_to_import() {
            NameToImport::Exact(exact_name) => {
                imports_locator::find_exact_imports(sema, current_crate, exact_name.clone())
            }
            // FIXME: ideally, we should avoid using `fst` for seacrhing trait imports for assoc items:
            // instead, we need to look up all trait impls for a certain struct and search through them only
            // see https://github.com/rust-analyzer/rust-analyzer/pull/7293#issuecomment-761585032
            // and https://rust-lang.zulipchat.com/#narrow/stream/185405-t-compiler.2Fwg-rls-2.2E0/topic/Blanket.20trait.20impls.20lookup
            // for the details
            NameToImport::Fuzzy(fuzzy_name) => {
                let (assoc_item_search, limit) = if self.import_candidate.is_trait_candidate() {
                    (AssocItemSearch::AssocItemsOnly, None)
                } else {
                    (AssocItemSearch::Include, Some(DEFAULT_QUERY_SEARCH_LIMIT))
                };

                imports_locator::find_similar_imports(
                    sema,
                    current_crate,
                    fuzzy_name.clone(),
                    assoc_item_search,
                    limit,
                )
            }
        };

        self.applicable_defs(sema.db, prefixed, defs_for_candidate_name)
            .into_iter()
            .filter(|import| import.import_path().len() > 1)
            .filter(|import| {
                let proposed_def = match import.item_to_import() {
                    ItemInNs::Types(id) => ScopeDef::ModuleDef(id.into()),
                    ItemInNs::Values(id) => ScopeDef::ModuleDef(id.into()),
                    ItemInNs::Macros(id) => ScopeDef::MacroDef(id.into()),
                };
                !scope_definitions.contains(&proposed_def)
            })
            .collect()
    }

    fn scope_definitions(&self) -> FxHashSet<ScopeDef> {
        let mut scope_definitions = FxHashSet::default();
        self.scope.process_all_names(&mut |_, scope_def| {
            scope_definitions.insert(scope_def);
        });
        scope_definitions
    }

    fn applicable_defs(
        &self,
        db: &RootDatabase,
        prefixed: Option<PrefixKind>,
        defs_for_candidate_name: impl Iterator<Item = Either<ModuleDef, MacroDef>>,
    ) -> FxHashSet<LocatedImport> {
        let current_crate = self.module_with_candidate.krate();

        let mod_path = |item| get_mod_path(db, item, &self.module_with_candidate, prefixed);

        match &self.import_candidate {
            ImportCandidate::Path(path_candidate) => {
                path_applicable_imports(db, path_candidate, mod_path, defs_for_candidate_name)
            }
            ImportCandidate::TraitAssocItem(trait_candidate) => trait_applicable_items(
                db,
                current_crate,
                trait_candidate,
                true,
                mod_path,
                defs_for_candidate_name,
            ),
            ImportCandidate::TraitMethod(trait_candidate) => trait_applicable_items(
                db,
                current_crate,
                trait_candidate,
                false,
                mod_path,
                defs_for_candidate_name,
            ),
        }
    }
}

fn path_applicable_imports(
    db: &RootDatabase,
    path_candidate: &PathImportCandidate,
    mod_path: impl Fn(ItemInNs) -> Option<ModPath> + Copy,
    defs_for_candidate_name: impl Iterator<Item = Either<ModuleDef, MacroDef>>,
) -> FxHashSet<LocatedImport> {
    let items_for_candidate_name =
        defs_for_candidate_name.map(|def| def.either(ItemInNs::from, ItemInNs::from));

    let (unresolved_first_segment, unresolved_qualifier) = match &path_candidate.qualifier {
        Qualifier::Absent => {
            return items_for_candidate_name
                .filter_map(|item| Some(LocatedImport::new(mod_path(item)?, item, None)))
                .collect();
        }
        Qualifier::FirstSegmentUnresolved(first_segment, qualifier) => {
            (first_segment.to_string(), qualifier.to_string())
        }
    };

    items_for_candidate_name
        .filter_map(|item| {
            import_for_item(db, mod_path, &unresolved_first_segment, &unresolved_qualifier, item)
        })
        .collect()
}

fn import_for_item(
    db: &RootDatabase,
    mod_path: impl Fn(ItemInNs) -> Option<ModPath>,
    unresolved_first_segment: &str,
    unresolved_qualifier: &str,
    original_item: ItemInNs,
) -> Option<LocatedImport> {
    let (item_candidate, trait_to_import) = match original_item {
        ItemInNs::Types(module_def_id) | ItemInNs::Values(module_def_id) => {
            match ModuleDef::from(module_def_id).as_assoc_item(db).map(|assoc| assoc.container(db))
            {
                Some(AssocItemContainer::Trait(trait_)) => {
                    let trait_item = ItemInNs::from(ModuleDef::from(trait_));
                    (trait_item, Some(trait_item))
                }
                Some(AssocItemContainer::Impl(impl_)) => {
                    (ItemInNs::from(ModuleDef::from(impl_.target_ty(db).as_adt()?)), None)
                }
                None => (original_item, None),
            }
        }
        ItemInNs::Macros(_) => (original_item, None),
    };
    let import_path_candidate = mod_path(item_candidate)?;

    let import_path_string = import_path_candidate.to_string();
    if !import_path_string.contains(unresolved_first_segment)
        || !import_path_string.contains(unresolved_qualifier)
    {
        return None;
    }

    let segment_import = find_import_for_segment(db, item_candidate, &unresolved_first_segment)?;
    let data_to_display = Some((import_path_candidate.clone(), original_item));
    Some(match (segment_import == item_candidate, trait_to_import) {
        (true, Some(_)) => {
            // FIXME we should be able to import both the trait and the segment,
            // but it's unclear what to do with overlapping edits (merge imports?)
            // especially in case of lazy completion edit resolutions.
            return None;
        }
        (false, Some(trait_to_import)) => {
            LocatedImport::new(mod_path(trait_to_import)?, trait_to_import, data_to_display)
        }
        (true, None) => LocatedImport::new(import_path_candidate, item_candidate, data_to_display),
        (false, None) => {
            LocatedImport::new(mod_path(segment_import)?, segment_import, data_to_display)
        }
    })
}

fn find_import_for_segment(
    db: &RootDatabase,
    original_item: ItemInNs,
    unresolved_first_segment: &str,
) -> Option<ItemInNs> {
    let segment_is_name = item_name(db, original_item)
        .map(|name| name.to_string() == unresolved_first_segment)
        .unwrap_or(false);

    Some(if segment_is_name {
        original_item
    } else {
        let matching_module =
            module_with_segment_name(db, &unresolved_first_segment, original_item)?;
        ItemInNs::from(ModuleDef::from(matching_module))
    })
}

fn module_with_segment_name(
    db: &RootDatabase,
    segment_name: &str,
    candidate: ItemInNs,
) -> Option<Module> {
    let mut current_module = match candidate {
        ItemInNs::Types(module_def_id) => ModuleDef::from(module_def_id).module(db),
        ItemInNs::Values(module_def_id) => ModuleDef::from(module_def_id).module(db),
        ItemInNs::Macros(macro_def_id) => MacroDef::from(macro_def_id).module(db),
    };
    while let Some(module) = current_module {
        if let Some(module_name) = module.name(db) {
            if module_name.to_string() == segment_name {
                return Some(module);
            }
        }
        current_module = module.parent(db);
    }
    None
}

fn trait_applicable_items(
    db: &RootDatabase,
    current_crate: Crate,
    trait_candidate: &TraitImportCandidate,
    trait_assoc_item: bool,
    mod_path: impl Fn(ItemInNs) -> Option<ModPath>,
    defs_for_candidate_name: impl Iterator<Item = Either<ModuleDef, MacroDef>>,
) -> FxHashSet<LocatedImport> {
    let mut required_assoc_items = FxHashSet::default();

    let trait_candidates = defs_for_candidate_name
        .filter_map(|input| match input {
            Either::Left(module_def) => module_def.as_assoc_item(db),
            _ => None,
        })
        .filter_map(|assoc| {
            let assoc_item_trait = assoc.containing_trait(db)?;
            required_assoc_items.insert(assoc);
            Some(assoc_item_trait.into())
        })
        .collect();

    let mut located_imports = FxHashSet::default();

    if trait_assoc_item {
        trait_candidate.receiver_ty.iterate_path_candidates(
            db,
            current_crate,
            &trait_candidates,
            None,
            |_, assoc| {
                if required_assoc_items.contains(&assoc) {
                    if let AssocItem::Function(f) = assoc {
                        if f.self_param(db).is_some() {
                            return None;
                        }
                    }

                    let item = ItemInNs::from(ModuleDef::from(assoc.containing_trait(db)?));
                    let item_path = mod_path(item)?;

                    let assoc_item = assoc_to_item(assoc);
                    let assoc_item_path = match assoc.container(db) {
                        AssocItemContainer::Trait(_) => item_path.clone(),
                        AssocItemContainer::Impl(impl_) => mod_path(ItemInNs::from(
                            ModuleDef::from(impl_.target_ty(db).as_adt()?),
                        ))?,
                    };

                    located_imports.insert(LocatedImport::new(
                        item_path,
                        item,
                        Some((assoc_item_path, assoc_item)),
                    ));
                }
                None::<()>
            },
        )
    } else {
        trait_candidate.receiver_ty.iterate_method_candidates(
            db,
            current_crate,
            &trait_candidates,
            None,
            |_, function| {
                let assoc = function.as_assoc_item(db)?;
                if required_assoc_items.contains(&assoc) {
                    let item = ItemInNs::from(ModuleDef::from(assoc.containing_trait(db)?));
                    let item_path = mod_path(item)?;

                    let assoc_item = assoc_to_item(assoc);
                    let assoc_item_path = match assoc.container(db) {
                        AssocItemContainer::Trait(_) => item_path.clone(),
                        AssocItemContainer::Impl(impl_) => mod_path(ItemInNs::from(
                            ModuleDef::from(impl_.target_ty(db).as_adt()?),
                        ))?,
                    };

                    located_imports.insert(LocatedImport::new(
                        item_path,
                        item,
                        Some((assoc_item_path, assoc_item)),
                    ));
                }
                None::<()>
            },
        )
    };

    located_imports
}

fn assoc_to_item(assoc: AssocItem) -> ItemInNs {
    match assoc {
        AssocItem::Function(f) => ItemInNs::from(ModuleDef::from(f)),
        AssocItem::Const(c) => ItemInNs::from(ModuleDef::from(c)),
        AssocItem::TypeAlias(t) => ItemInNs::from(ModuleDef::from(t)),
    }
}

fn get_mod_path(
    db: &RootDatabase,
    item_to_search: ItemInNs,
    module_with_candidate: &Module,
    prefixed: Option<PrefixKind>,
) -> Option<ModPath> {
    if let Some(prefix_kind) = prefixed {
        module_with_candidate.find_use_path_prefixed(db, item_to_search, prefix_kind)
    } else {
        module_with_candidate.find_use_path(db, item_to_search)
    }
}

impl ImportCandidate {
    fn for_method_call(
        sema: &Semantics<RootDatabase>,
        method_call: &ast::MethodCallExpr,
    ) -> Option<Self> {
        match sema.resolve_method_call(method_call) {
            Some(_) => None,
            None => Some(Self::TraitMethod(TraitImportCandidate {
                receiver_ty: sema.type_of_expr(&method_call.receiver()?)?,
                name: NameToImport::Exact(method_call.name_ref()?.to_string()),
            })),
        }
    }

    fn for_regular_path(sema: &Semantics<RootDatabase>, path: &ast::Path) -> Option<Self> {
        if sema.resolve_path(path).is_some() {
            return None;
        }
        path_import_candidate(
            sema,
            path.qualifier(),
            NameToImport::Exact(path.segment()?.name_ref()?.to_string()),
        )
    }

    fn for_fuzzy_path(
        qualifier: Option<ast::Path>,
        fuzzy_name: String,
        sema: &Semantics<RootDatabase>,
    ) -> Option<Self> {
        path_import_candidate(sema, qualifier, NameToImport::Fuzzy(fuzzy_name))
    }

    fn is_trait_candidate(&self) -> bool {
        matches!(self, ImportCandidate::TraitAssocItem(_) | ImportCandidate::TraitMethod(_))
    }
}

fn path_import_candidate(
    sema: &Semantics<RootDatabase>,
    qualifier: Option<ast::Path>,
    name: NameToImport,
) -> Option<ImportCandidate> {
    Some(match qualifier {
        Some(qualifier) => match sema.resolve_path(&qualifier) {
            None => {
                let qualifier_start =
                    qualifier.syntax().descendants().find_map(ast::NameRef::cast)?;
                let qualifier_start_path =
                    qualifier_start.syntax().ancestors().find_map(ast::Path::cast)?;
                if sema.resolve_path(&qualifier_start_path).is_none() {
                    ImportCandidate::Path(PathImportCandidate {
                        qualifier: Qualifier::FirstSegmentUnresolved(
                            qualifier_start,
                            ModPath::from_src_unhygienic(qualifier)?,
                        ),
                        name,
                    })
                } else {
                    return None;
                }
            }
            Some(PathResolution::Def(ModuleDef::Adt(assoc_item_path))) => {
                ImportCandidate::TraitAssocItem(TraitImportCandidate {
                    receiver_ty: assoc_item_path.ty(sema.db),
                    name,
                })
            }
            Some(_) => return None,
        },
        None => ImportCandidate::Path(PathImportCandidate { qualifier: Qualifier::Absent, name }),
    })
}
