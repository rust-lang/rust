//! Look up accessible paths for items.
use either::Either;
use hir::{AsAssocItem, AssocItem, Crate, MacroDef, Module, ModuleDef, PrefixKind, Semantics};
use rustc_hash::FxHashSet;
use syntax::{ast, AstNode};

use crate::{
    imports_locator::{self, AssocItemSearch, DEFAULT_QUERY_SEARCH_LIMIT},
    RootDatabase,
};

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
    pub receiver_ty: hir::Type,
    pub name: NameToImport,
}

#[derive(Debug)]
pub struct PathImportCandidate {
    pub qualifier: Option<ast::Path>,
    pub name: NameToImport,
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
    module_with_candidate: hir::Module,
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
        module_with_path: Module,
        qualifier: Option<ast::Path>,
        fuzzy_name: String,
        sema: &Semantics<RootDatabase>,
    ) -> Option<Self> {
        Some(match qualifier {
            Some(qualifier) => {
                let qualifier_resolution = sema.resolve_path(&qualifier)?;
                match qualifier_resolution {
                    hir::PathResolution::Def(hir::ModuleDef::Adt(assoc_item_path)) => Self {
                        import_candidate: ImportCandidate::TraitAssocItem(TraitImportCandidate {
                            receiver_ty: assoc_item_path.ty(sema.db),
                            name: NameToImport::Fuzzy(fuzzy_name),
                        }),
                        module_with_candidate: module_with_path,
                    },
                    _ => Self {
                        import_candidate: ImportCandidate::Path(PathImportCandidate {
                            qualifier: Some(qualifier),
                            name: NameToImport::Fuzzy(fuzzy_name),
                        }),
                        module_with_candidate: module_with_path,
                    },
                }
            }
            None => Self {
                import_candidate: ImportCandidate::Path(PathImportCandidate {
                    qualifier: None,
                    name: NameToImport::Fuzzy(fuzzy_name),
                }),
                module_with_candidate: module_with_path,
            },
        })
    }

    pub fn for_fuzzy_method_call(
        module_with_method_call: Module,
        receiver_ty: hir::Type,
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
    ) -> Vec<(hir::ModPath, hir::ItemInNs)> {
        let _p = profile::span("import_assets::search_for_imports");
        self.search_for(sema, Some(prefix_kind))
    }

    /// This may return non-absolute paths if a part of the returned path is already imported into scope.
    pub fn search_for_relative_paths(
        &self,
        sema: &Semantics<RootDatabase>,
    ) -> Vec<(hir::ModPath, hir::ItemInNs)> {
        let _p = profile::span("import_assets::search_for_relative_paths");
        self.search_for(sema, None)
    }

    fn search_for(
        &self,
        sema: &Semantics<RootDatabase>,
        prefixed: Option<hir::PrefixKind>,
    ) -> Vec<(hir::ModPath, hir::ItemInNs)> {
        let current_crate = self.module_with_candidate.krate();

        let unfiltered_imports = match self.name_to_import() {
            NameToImport::Exact(exact_name) => {
                imports_locator::find_exact_imports(sema, current_crate, exact_name.clone())
            }
            // FIXME: ideally, we should avoid using `fst` for seacrhing trait imports for assoc items:
            // instead, we need to look up all trait impls for a certain struct and search through them only
            // see https://github.com/rust-analyzer/rust-analyzer/pull/7293#issuecomment-761585032
            // and https://rust-lang.zulipchat.com/#narrow/stream/185405-t-compiler.2Fwg-rls-2.2E0/topic/Blanket.20trait.20impls.20lookup
            // for the details
            NameToImport::Fuzzy(fuzzy_name) => {
                let (assoc_item_search, limit) = match self.import_candidate {
                    ImportCandidate::TraitAssocItem(_) | ImportCandidate::TraitMethod(_) => {
                        (AssocItemSearch::AssocItemsOnly, None)
                    }
                    _ => (AssocItemSearch::Exclude, Some(DEFAULT_QUERY_SEARCH_LIMIT)),
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

        let db = sema.db;
        let mut res =
            applicable_defs(self.import_candidate(), current_crate, db, unfiltered_imports)
                .filter_map(|candidate| {
                    let item: hir::ItemInNs = candidate.clone().either(Into::into, Into::into);

                    let item_to_search = match self.import_candidate {
                        ImportCandidate::TraitAssocItem(_) | ImportCandidate::TraitMethod(_) => {
                            let canidate_trait = match candidate {
                                Either::Left(module_def) => {
                                    module_def.as_assoc_item(db)?.containing_trait(db)
                                }
                                _ => None,
                            }?;
                            ModuleDef::from(canidate_trait).into()
                        }
                        _ => item,
                    };

                    if let Some(prefix_kind) = prefixed {
                        self.module_with_candidate.find_use_path_prefixed(
                            db,
                            item_to_search,
                            prefix_kind,
                        )
                    } else {
                        self.module_with_candidate.find_use_path(db, item_to_search)
                    }
                    .map(|path| (path, item))
                })
                .filter(|(use_path, _)| use_path.len() > 1)
                .collect::<Vec<_>>();
        res.sort_by_cached_key(|(path, _)| path.clone());
        res
    }
}

fn applicable_defs<'a>(
    import_candidate: &ImportCandidate,
    current_crate: Crate,
    db: &RootDatabase,
    unfiltered_imports: Box<dyn Iterator<Item = Either<ModuleDef, MacroDef>> + 'a>,
) -> Box<dyn Iterator<Item = Either<ModuleDef, MacroDef>> + 'a> {
    let receiver_ty = match import_candidate {
        ImportCandidate::Path(_) => return unfiltered_imports,
        ImportCandidate::TraitAssocItem(candidate) | ImportCandidate::TraitMethod(candidate) => {
            &candidate.receiver_ty
        }
    };

    let mut required_assoc_items = FxHashSet::default();

    let trait_candidates = unfiltered_imports
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

    let mut applicable_defs = FxHashSet::default();

    match import_candidate {
        ImportCandidate::Path(_) => unreachable!(),
        ImportCandidate::TraitAssocItem(_) => receiver_ty.iterate_path_candidates(
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
                    applicable_defs.insert(Either::Left(assoc_to_module_def(assoc)));
                }
                None::<()>
            },
        ),
        ImportCandidate::TraitMethod(_) => receiver_ty.iterate_method_candidates(
            db,
            current_crate,
            &trait_candidates,
            None,
            |_, function| {
                let assoc = function.as_assoc_item(db)?;
                if required_assoc_items.contains(&assoc) {
                    applicable_defs.insert(Either::Left(assoc_to_module_def(assoc)));
                }
                None::<()>
            },
        ),
    };

    Box::new(applicable_defs.into_iter())
}

fn assoc_to_module_def(assoc: AssocItem) -> ModuleDef {
    match assoc {
        AssocItem::Function(f) => f.into(),
        AssocItem::Const(c) => c.into(),
        AssocItem::TypeAlias(t) => t.into(),
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

        let segment = path.segment()?;
        let candidate = if let Some(qualifier) = path.qualifier() {
            let qualifier_start = qualifier.syntax().descendants().find_map(ast::NameRef::cast)?;
            let qualifier_start_path =
                qualifier_start.syntax().ancestors().find_map(ast::Path::cast)?;
            if let Some(qualifier_start_resolution) = sema.resolve_path(&qualifier_start_path) {
                let qualifier_resolution = if qualifier_start_path == qualifier {
                    qualifier_start_resolution
                } else {
                    sema.resolve_path(&qualifier)?
                };
                match qualifier_resolution {
                    hir::PathResolution::Def(hir::ModuleDef::Adt(assoc_item_path)) => {
                        ImportCandidate::TraitAssocItem(TraitImportCandidate {
                            receiver_ty: assoc_item_path.ty(sema.db),
                            name: NameToImport::Exact(segment.name_ref()?.to_string()),
                        })
                    }
                    _ => return None,
                }
            } else {
                ImportCandidate::Path(PathImportCandidate {
                    qualifier: Some(qualifier),
                    name: NameToImport::Exact(qualifier_start.to_string()),
                })
            }
        } else {
            ImportCandidate::Path(PathImportCandidate {
                qualifier: None,
                name: NameToImport::Exact(
                    segment.syntax().descendants().find_map(ast::NameRef::cast)?.to_string(),
                ),
            })
        };
        Some(candidate)
    }
}
