//! Look up accessible paths for items.
use std::collections::BTreeSet;

use either::Either;
use hir::{AsAssocItem, AssocItemContainer, ModuleDef, Semantics};
use ide_db::{imports_locator, RootDatabase};
use rustc_hash::FxHashSet;
use syntax::{ast, AstNode, SyntaxNode};

use crate::assist_config::InsertUseConfig;

#[derive(Debug)]
pub(crate) enum ImportCandidate {
    /// Simple name like 'HashMap'
    UnqualifiedName(String),
    /// First part of the qualified name.
    /// For 'std::collections::HashMap', that will be 'std'.
    QualifierStart(String),
    /// A trait associated function (with no self parameter) or associated constant.
    /// For 'test_mod::TestEnum::test_function', `Type` is the `test_mod::TestEnum` expression type
    /// and `String` is the `test_function`
    TraitAssocItem(hir::Type, String),
    /// A trait method with self parameter.
    /// For 'test_enum.test_method()', `Type` is the `test_enum` expression type
    /// and `String` is the `test_method`
    TraitMethod(hir::Type, String),
}

#[derive(Debug)]
pub(crate) struct ImportAssets {
    import_candidate: ImportCandidate,
    module_with_name_to_import: hir::Module,
    syntax_under_caret: SyntaxNode,
}

impl ImportAssets {
    pub(crate) fn for_method_call(
        method_call: ast::MethodCallExpr,
        sema: &Semantics<RootDatabase>,
    ) -> Option<Self> {
        let syntax_under_caret = method_call.syntax().to_owned();
        let module_with_name_to_import = sema.scope(&syntax_under_caret).module()?;
        Some(Self {
            import_candidate: ImportCandidate::for_method_call(sema, &method_call)?,
            module_with_name_to_import,
            syntax_under_caret,
        })
    }

    pub(crate) fn for_regular_path(
        path_under_caret: ast::Path,
        sema: &Semantics<RootDatabase>,
    ) -> Option<Self> {
        let syntax_under_caret = path_under_caret.syntax().to_owned();
        if syntax_under_caret.ancestors().find_map(ast::Use::cast).is_some() {
            return None;
        }

        let module_with_name_to_import = sema.scope(&syntax_under_caret).module()?;
        Some(Self {
            import_candidate: ImportCandidate::for_regular_path(sema, &path_under_caret)?,
            module_with_name_to_import,
            syntax_under_caret,
        })
    }

    pub(crate) fn syntax_under_caret(&self) -> &SyntaxNode {
        &self.syntax_under_caret
    }

    pub(crate) fn import_candidate(&self) -> &ImportCandidate {
        &self.import_candidate
    }

    fn get_search_query(&self) -> &str {
        match &self.import_candidate {
            ImportCandidate::UnqualifiedName(name) => name,
            ImportCandidate::QualifierStart(qualifier_start) => qualifier_start,
            ImportCandidate::TraitAssocItem(_, trait_assoc_item_name) => trait_assoc_item_name,
            ImportCandidate::TraitMethod(_, trait_method_name) => trait_method_name,
        }
    }

    pub(crate) fn search_for_imports(
        &self,
        sema: &Semantics<RootDatabase>,
        config: &InsertUseConfig,
    ) -> BTreeSet<hir::ModPath> {
        let _p = profile::span("import_assists::search_for_imports");
        self.search_for(sema, Some(config.prefix_kind))
    }

    /// This may return non-absolute paths if a part of the returned path is already imported into scope.
    #[allow(dead_code)]
    pub(crate) fn search_for_relative_paths(
        &self,
        sema: &Semantics<RootDatabase>,
    ) -> BTreeSet<hir::ModPath> {
        let _p = profile::span("import_assists::search_for_relative_paths");
        self.search_for(sema, None)
    }

    fn search_for(
        &self,
        sema: &Semantics<RootDatabase>,
        prefixed: Option<hir::PrefixKind>,
    ) -> BTreeSet<hir::ModPath> {
        let db = sema.db;
        let current_crate = self.module_with_name_to_import.krate();
        imports_locator::find_imports(sema, current_crate, &self.get_search_query())
            .into_iter()
            .filter_map(|candidate| match &self.import_candidate {
                ImportCandidate::TraitAssocItem(assoc_item_type, _) => {
                    let located_assoc_item = match candidate {
                        Either::Left(ModuleDef::Function(located_function)) => located_function
                            .as_assoc_item(db)
                            .map(|assoc| assoc.container(db))
                            .and_then(Self::assoc_to_trait),
                        Either::Left(ModuleDef::Const(located_const)) => located_const
                            .as_assoc_item(db)
                            .map(|assoc| assoc.container(db))
                            .and_then(Self::assoc_to_trait),
                        _ => None,
                    }?;

                    let mut trait_candidates = FxHashSet::default();
                    trait_candidates.insert(located_assoc_item.into());

                    assoc_item_type
                        .iterate_path_candidates(
                            db,
                            current_crate,
                            &trait_candidates,
                            None,
                            |_, assoc| Self::assoc_to_trait(assoc.container(db)),
                        )
                        .map(ModuleDef::from)
                        .map(Either::Left)
                }
                ImportCandidate::TraitMethod(function_callee, _) => {
                    let located_assoc_item =
                        if let Either::Left(ModuleDef::Function(located_function)) = candidate {
                            located_function
                                .as_assoc_item(db)
                                .map(|assoc| assoc.container(db))
                                .and_then(Self::assoc_to_trait)
                        } else {
                            None
                        }?;

                    let mut trait_candidates = FxHashSet::default();
                    trait_candidates.insert(located_assoc_item.into());

                    function_callee
                        .iterate_method_candidates(
                            db,
                            current_crate,
                            &trait_candidates,
                            None,
                            |_, function| {
                                Self::assoc_to_trait(function.as_assoc_item(db)?.container(db))
                            },
                        )
                        .map(ModuleDef::from)
                        .map(Either::Left)
                }
                _ => Some(candidate),
            })
            .filter_map(|candidate| {
                let item: hir::ItemInNs = match candidate {
                    Either::Left(module_def) => module_def.into(),
                    Either::Right(macro_def) => macro_def.into(),
                };
                if let Some(prefix_kind) = prefixed {
                    self.module_with_name_to_import.find_use_path_prefixed(db, item, prefix_kind)
                } else {
                    self.module_with_name_to_import.find_use_path(db, item)
                }
            })
            .filter(|use_path| !use_path.segments.is_empty())
            .take(20)
            .collect::<BTreeSet<_>>()
    }

    fn assoc_to_trait(assoc: AssocItemContainer) -> Option<hir::Trait> {
        if let AssocItemContainer::Trait(extracted_trait) = assoc {
            Some(extracted_trait)
        } else {
            None
        }
    }
}

impl ImportCandidate {
    fn for_method_call(
        sema: &Semantics<RootDatabase>,
        method_call: &ast::MethodCallExpr,
    ) -> Option<Self> {
        if sema.resolve_method_call(method_call).is_some() {
            return None;
        }
        Some(Self::TraitMethod(
            sema.type_of_expr(&method_call.receiver()?)?,
            method_call.name_ref()?.syntax().to_string(),
        ))
    }

    fn for_regular_path(
        sema: &Semantics<RootDatabase>,
        path_under_caret: &ast::Path,
    ) -> Option<Self> {
        if sema.resolve_path(path_under_caret).is_some() {
            return None;
        }

        let segment = path_under_caret.segment()?;
        if let Some(qualifier) = path_under_caret.qualifier() {
            let qualifier_start = qualifier.syntax().descendants().find_map(ast::NameRef::cast)?;
            let qualifier_start_path =
                qualifier_start.syntax().ancestors().find_map(ast::Path::cast)?;
            if let Some(qualifier_start_resolution) = sema.resolve_path(&qualifier_start_path) {
                let qualifier_resolution = if qualifier_start_path == qualifier {
                    qualifier_start_resolution
                } else {
                    sema.resolve_path(&qualifier)?
                };
                if let hir::PathResolution::Def(hir::ModuleDef::Adt(assoc_item_path)) =
                    qualifier_resolution
                {
                    Some(ImportCandidate::TraitAssocItem(
                        assoc_item_path.ty(sema.db),
                        segment.syntax().to_string(),
                    ))
                } else {
                    None
                }
            } else {
                Some(ImportCandidate::QualifierStart(qualifier_start.syntax().to_string()))
            }
        } else {
            Some(ImportCandidate::UnqualifiedName(
                segment.syntax().descendants().find_map(ast::NameRef::cast)?.syntax().to_string(),
            ))
        }
    }
}
