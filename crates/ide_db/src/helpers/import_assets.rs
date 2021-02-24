//! Look up accessible paths for items.
use either::Either;
use hir::{
    AsAssocItem, AssocItem, AssocItemContainer, Crate, ItemInNs, MacroDef, ModPath, Module,
    ModuleDef, PathResolution, PrefixKind, Semantics, Type,
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
    FirstSegmentUnresolved(ast::PathSegment, ast::Path),
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
pub struct ImportAssets {
    import_candidate: ImportCandidate,
    module_with_candidate: Module,
}

impl ImportAssets {
    pub fn for_method_call(
        method_call: &ast::MethodCallExpr,
        sema: &Semantics<RootDatabase>,
    ) -> Option<Self> {
        Some(Self {
            import_candidate: ImportCandidate::for_method_call(sema, method_call)?,
            module_with_candidate: sema.scope(method_call.syntax()).module()?,
        })
    }

    pub fn for_exact_path(
        fully_qualified_path: &ast::Path,
        sema: &Semantics<RootDatabase>,
    ) -> Option<Self> {
        let syntax_under_caret = fully_qualified_path.syntax();
        if syntax_under_caret.ancestors().find_map(ast::Use::cast).is_some() {
            return None;
        }
        Some(Self {
            import_candidate: ImportCandidate::for_regular_path(sema, fully_qualified_path)?,
            module_with_candidate: sema.scope(syntax_under_caret).module()?,
        })
    }

    pub fn for_fuzzy_path(
        module_with_candidate: Module,
        qualifier: Option<ast::Path>,
        fuzzy_name: String,
        sema: &Semantics<RootDatabase>,
    ) -> Option<Self> {
        Some(Self {
            import_candidate: ImportCandidate::for_fuzzy_path(qualifier, fuzzy_name, sema)?,
            module_with_candidate,
        })
    }

    pub fn for_fuzzy_method_call(
        module_with_method_call: Module,
        receiver_ty: Type,
        fuzzy_method_name: String,
    ) -> Option<Self> {
        Some(Self {
            import_candidate: ImportCandidate::TraitMethod(TraitImportCandidate {
                receiver_ty,
                name: NameToImport::Fuzzy(fuzzy_method_name),
            }),
            module_with_candidate: module_with_method_call,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LocatedImport {
    import_path: ModPath,
    item_to_import: ItemInNs,
    import_display_override: Option<(ModPath, ItemInNs)>,
}

impl LocatedImport {
    pub fn new(
        import_path: ModPath,
        item_to_import: ItemInNs,
        import_display_override: Option<(ModPath, ItemInNs)>,
    ) -> Self {
        Self { import_path, item_to_import, import_display_override }
    }

    pub fn display_path(&self) -> &ModPath {
        self.import_display_override
            .as_ref()
            .map(|(mod_path, _)| mod_path)
            .unwrap_or(&self.import_path)
    }

    pub fn import_path(&self) -> &ModPath {
        &self.import_path
    }

    pub fn item_to_display(&self) -> ItemInNs {
        self.import_display_override.as_ref().map(|&(_, item)| item).unwrap_or(self.item_to_import)
    }

    pub fn item_to_import(&self) -> ItemInNs {
        self.item_to_import
    }
}

impl ImportAssets {
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

        let imports_for_candidate_name = match self.name_to_import() {
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

        self.applicable_defs(sema.db, prefixed, imports_for_candidate_name)
            .into_iter()
            .filter(|import| import.import_path().len() > 1)
            .collect()
    }

    fn applicable_defs(
        &self,
        db: &RootDatabase,
        prefixed: Option<PrefixKind>,
        unfiltered_defs: impl Iterator<Item = Either<ModuleDef, MacroDef>>,
    ) -> FxHashSet<LocatedImport> {
        let current_crate = self.module_with_candidate.krate();

        let import_path_locator =
            |item| get_mod_path(db, item, &self.module_with_candidate, prefixed);

        match &self.import_candidate {
            ImportCandidate::Path(path_candidate) => {
                path_applicable_imports(db, path_candidate, import_path_locator, unfiltered_defs)
            }
            ImportCandidate::TraitAssocItem(trait_candidate) => trait_applicable_items(
                db,
                current_crate,
                trait_candidate,
                true,
                import_path_locator,
                unfiltered_defs,
            ),
            ImportCandidate::TraitMethod(trait_candidate) => trait_applicable_items(
                db,
                current_crate,
                trait_candidate,
                false,
                import_path_locator,
                unfiltered_defs,
            ),
        }
    }
}

fn path_applicable_imports(
    db: &RootDatabase,
    path_candidate: &PathImportCandidate,
    import_path_locator: impl Fn(ItemInNs) -> Option<ModPath>,
    unfiltered_defs: impl Iterator<Item = Either<ModuleDef, MacroDef>>,
) -> FxHashSet<LocatedImport> {
    let applicable_items = unfiltered_defs
        .filter_map(|def| {
            let (assoc_original, candidate) = match def {
                Either::Left(module_def) => match module_def.as_assoc_item(db) {
                    Some(assoc_item) => match assoc_item.container(db) {
                        AssocItemContainer::Trait(trait_) => {
                            (Some(module_def), ItemInNs::from(ModuleDef::from(trait_)))
                        }
                        AssocItemContainer::Impl(impl_) => (
                            Some(module_def),
                            ItemInNs::from(ModuleDef::from(impl_.target_ty(db).as_adt()?)),
                        ),
                    },
                    None => (None, ItemInNs::from(module_def)),
                },
                Either::Right(macro_def) => (None, ItemInNs::from(macro_def)),
            };
            Some((assoc_original, candidate))
        })
        .filter_map(|(assoc_original, candidate)| {
            import_path_locator(candidate).zip(Some((assoc_original, candidate)))
        });

    let (unresolved_first_segment, unresolved_qualifier) = match &path_candidate.qualifier {
        Qualifier::Absent => {
            return applicable_items
                .map(|(candidate_path, (_, candidate))| {
                    LocatedImport::new(candidate_path, candidate, None)
                })
                .collect();
        }
        Qualifier::FirstSegmentUnresolved(first_segment, qualifier) => (first_segment, qualifier),
    };

    // TODO kb need to remove turbofish from the qualifier, maybe use the segments instead?
    // TODO kb sorting is changed now, return back?
    let unresolved_qualifier_string = unresolved_qualifier.to_string();
    let unresolved_first_segment_string = unresolved_first_segment.to_string();

    applicable_items
        .filter(|(candidate_path, _)| {
            let candidate_path_string = candidate_path.to_string();
            candidate_path_string.contains(&unresolved_qualifier_string)
                && candidate_path_string.contains(&unresolved_first_segment_string)
        })
        .filter_map(|(candidate_path, (assoc_original, candidate))| {
            let found_segment_resolution = item_name(db, candidate)
                .map(|name| name.to_string() == unresolved_first_segment_string)
                .unwrap_or(false);
            let (import_path, item_to_import) = if found_segment_resolution {
                (candidate_path.clone(), candidate)
            } else {
                let matching_module =
                    module_with_matching_name(db, &unresolved_first_segment_string, candidate)?;
                let module_item = ItemInNs::from(ModuleDef::from(matching_module));
                (import_path_locator(module_item)?, module_item)
            };

            Some(match assoc_original {
                Some(assoc_original) => LocatedImport::new(
                    import_path.clone(),
                    item_to_import,
                    Some((import_path, ItemInNs::from(assoc_original))),
                ),
                None => LocatedImport::new(
                    import_path,
                    item_to_import,
                    if found_segment_resolution { None } else { Some((candidate_path, candidate)) },
                ),
            })
        })
        .collect()
}

fn item_module(db: &RootDatabase, item: ItemInNs) -> Option<Module> {
    match item {
        ItemInNs::Types(module_def_id) => ModuleDef::from(module_def_id).module(db),
        ItemInNs::Values(module_def_id) => ModuleDef::from(module_def_id).module(db),
        ItemInNs::Macros(macro_def_id) => MacroDef::from(macro_def_id).module(db),
    }
}

fn module_with_matching_name(
    db: &RootDatabase,
    unresolved_first_segment_string: &str,
    candidate: ItemInNs,
) -> Option<Module> {
    let mut current_module = item_module(db, candidate);
    while let Some(module) = current_module {
        match module.name(db) {
            Some(module_name) => {
                if module_name.to_string().as_str() == unresolved_first_segment_string {
                    return Some(module);
                }
            }
            None => {}
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
    import_path_locator: impl Fn(ItemInNs) -> Option<ModPath>,
    unfiltered_defs: impl Iterator<Item = Either<ModuleDef, MacroDef>>,
) -> FxHashSet<LocatedImport> {
    let mut required_assoc_items = FxHashSet::default();

    let trait_candidates = unfiltered_defs
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
                    let item_path = import_path_locator(item)?;

                    let assoc_item = assoc_to_item(assoc);
                    let assoc_item_path = match assoc.container(db) {
                        AssocItemContainer::Trait(_) => item_path.clone(),
                        AssocItemContainer::Impl(impl_) => import_path_locator(ItemInNs::from(
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
                    let item_path = import_path_locator(item)?;

                    let assoc_item = assoc_to_item(assoc);
                    let assoc_item_path = match assoc.container(db) {
                        AssocItemContainer::Trait(_) => item_path.clone(),
                        AssocItemContainer::Impl(impl_) => import_path_locator(ItemInNs::from(
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
                    qualifier.syntax().descendants().find_map(ast::PathSegment::cast)?;
                let qualifier_start_path =
                    qualifier_start.syntax().ancestors().find_map(ast::Path::cast)?;
                if sema.resolve_path(&qualifier_start_path).is_none() {
                    ImportCandidate::Path(PathImportCandidate {
                        qualifier: Qualifier::FirstSegmentUnresolved(qualifier_start, qualifier),
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
