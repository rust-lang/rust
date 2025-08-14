//! Look up accessible paths for items.

use std::ops::ControlFlow;

use hir::{
    AsAssocItem, AssocItem, AssocItemContainer, Complete, Crate, HasCrate, ImportPathConfig,
    ItemInNs, ModPath, Module, ModuleDef, Name, PathResolution, PrefixKind, ScopeDef, Semantics,
    SemanticsScope, Trait, TyFingerprint, Type, db::HirDatabase,
};
use itertools::Itertools;
use rustc_hash::{FxHashMap, FxHashSet};
use syntax::{
    AstNode, SyntaxNode,
    ast::{self, HasName, make},
};

use crate::{
    FxIndexSet, RootDatabase,
    items_locator::{self, AssocSearchMode, DEFAULT_QUERY_SEARCH_LIMIT},
};

/// A candidate for import, derived during various IDE activities:
/// * completion with imports on the fly proposals
/// * completion edit resolve requests
/// * assists
/// * etc.
#[derive(Debug)]
pub enum ImportCandidate<'db> {
    /// A path, qualified (`std::collections::HashMap`) or not (`HashMap`).
    Path(PathImportCandidate),
    /// A trait associated function (with no self parameter) or an associated constant.
    /// For 'test_mod::TestEnum::test_function', `ty` is the `test_mod::TestEnum` expression type
    /// and `name` is the `test_function`
    TraitAssocItem(TraitImportCandidate<'db>),
    /// A trait method with self parameter.
    /// For 'test_enum.test_method()', `ty` is the `test_enum` expression type
    /// and `name` is the `test_method`
    TraitMethod(TraitImportCandidate<'db>),
}

/// A trait import needed for a given associated item access.
/// For `some::path::SomeStruct::ASSOC_`, contains the
/// type of `some::path::SomeStruct` and `ASSOC_` as the item name.
#[derive(Debug)]
pub struct TraitImportCandidate<'db> {
    /// A type of the item that has the associated item accessed at.
    pub receiver_ty: Type<'db>,
    /// The associated item name that the trait to import should contain.
    pub assoc_item_name: NameToImport,
}

/// Path import for a given name, qualified or not.
#[derive(Debug)]
pub struct PathImportCandidate {
    /// Optional qualifier before name.
    pub qualifier: Vec<Name>,
    /// The name the item (struct, trait, enum, etc.) should have.
    pub name: NameToImport,
}

/// A name that will be used during item lookups.
#[derive(Debug, Clone)]
pub enum NameToImport {
    /// Requires items with names that exactly match the given string, bool indicates case-sensitivity.
    Exact(String, bool),
    /// Requires items with names that match the given string by prefix, bool indicates case-sensitivity.
    Prefix(String, bool),
    /// Requires items with names contain all letters from the string,
    /// in the same order, but not necessary adjacent.
    Fuzzy(String, bool),
}

impl NameToImport {
    pub fn exact_case_sensitive(s: String) -> NameToImport {
        let s = match s.strip_prefix("r#") {
            Some(s) => s.to_owned(),
            None => s,
        };
        NameToImport::Exact(s, true)
    }

    pub fn fuzzy(s: String) -> NameToImport {
        let s = match s.strip_prefix("r#") {
            Some(s) => s.to_owned(),
            None => s,
        };
        // unless all chars are lowercase, we do a case sensitive search
        let case_sensitive = s.chars().any(|c| c.is_uppercase());
        NameToImport::Fuzzy(s, case_sensitive)
    }

    pub fn text(&self) -> &str {
        match self {
            NameToImport::Prefix(text, _)
            | NameToImport::Exact(text, _)
            | NameToImport::Fuzzy(text, _) => text.as_str(),
        }
    }
}

/// A struct to find imports in the project, given a certain name (or its part) and the context.
#[derive(Debug)]
pub struct ImportAssets<'db> {
    import_candidate: ImportCandidate<'db>,
    candidate_node: SyntaxNode,
    module_with_candidate: Module,
}

impl<'db> ImportAssets<'db> {
    pub fn for_method_call(
        method_call: &ast::MethodCallExpr,
        sema: &Semantics<'db, RootDatabase>,
    ) -> Option<Self> {
        let candidate_node = method_call.syntax().clone();
        Some(Self {
            import_candidate: ImportCandidate::for_method_call(sema, method_call)?,
            module_with_candidate: sema.scope(&candidate_node)?.module(),
            candidate_node,
        })
    }

    pub fn for_exact_path(
        fully_qualified_path: &ast::Path,
        sema: &Semantics<'db, RootDatabase>,
    ) -> Option<Self> {
        let candidate_node = fully_qualified_path.syntax().clone();
        if let Some(use_tree) = candidate_node.ancestors().find_map(ast::UseTree::cast) {
            // Path is inside a use tree, then only continue if it is the first segment of a use statement.
            if use_tree.syntax().parent().and_then(ast::Use::cast).is_none()
                || fully_qualified_path.qualifier().is_some()
            {
                return None;
            }
        }
        Some(Self {
            import_candidate: ImportCandidate::for_regular_path(sema, fully_qualified_path)?,
            module_with_candidate: sema.scope(&candidate_node)?.module(),
            candidate_node,
        })
    }

    pub fn for_ident_pat(sema: &Semantics<'db, RootDatabase>, pat: &ast::IdentPat) -> Option<Self> {
        if !pat.is_simple_ident() {
            return None;
        }
        let name = pat.name()?;
        let candidate_node = pat.syntax().clone();
        Some(Self {
            import_candidate: ImportCandidate::for_name(sema, &name)?,
            module_with_candidate: sema.scope(&candidate_node)?.module(),
            candidate_node,
        })
    }

    pub fn for_fuzzy_path(
        module_with_candidate: Module,
        qualifier: Option<ast::Path>,
        fuzzy_name: String,
        sema: &Semantics<'db, RootDatabase>,
        candidate_node: SyntaxNode,
    ) -> Option<Self> {
        Some(Self {
            import_candidate: ImportCandidate::for_fuzzy_path(qualifier, fuzzy_name, sema)?,
            module_with_candidate,
            candidate_node,
        })
    }

    pub fn for_fuzzy_method_call(
        module_with_method_call: Module,
        receiver_ty: Type<'db>,
        fuzzy_method_name: String,
        candidate_node: SyntaxNode,
    ) -> Option<Self> {
        Some(Self {
            import_candidate: ImportCandidate::TraitMethod(TraitImportCandidate {
                receiver_ty,
                assoc_item_name: NameToImport::fuzzy(fuzzy_method_name),
            }),
            module_with_candidate: module_with_method_call,
            candidate_node,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompleteInFlyimport(pub bool);

/// An import (not necessary the only one) that corresponds a certain given [`PathImportCandidate`].
/// (the structure is not entirely correct, since there can be situations requiring two imports, see FIXME below for the details)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LocatedImport {
    /// The path to use in the `use` statement for a given candidate to be imported.
    pub import_path: ModPath,
    /// An item that will be imported with the import path given.
    pub item_to_import: ItemInNs,
    /// The path import candidate, resolved.
    ///
    /// Not necessary matches the import:
    /// For any associated constant from the trait, we try to access as `some::path::SomeStruct::ASSOC_`
    /// the original item is the associated constant, but the import has to be a trait that
    /// defines this constant.
    pub original_item: ItemInNs,
    /// The value of `#[rust_analyzer::completions(...)]`, if existing.
    pub complete_in_flyimport: CompleteInFlyimport,
}

impl LocatedImport {
    pub fn new(
        import_path: ModPath,
        item_to_import: ItemInNs,
        original_item: ItemInNs,
        complete_in_flyimport: CompleteInFlyimport,
    ) -> Self {
        Self { import_path, item_to_import, original_item, complete_in_flyimport }
    }

    pub fn new_no_completion(
        import_path: ModPath,
        item_to_import: ItemInNs,
        original_item: ItemInNs,
    ) -> Self {
        Self {
            import_path,
            item_to_import,
            original_item,
            complete_in_flyimport: CompleteInFlyimport(true),
        }
    }
}

impl<'db> ImportAssets<'db> {
    pub fn import_candidate(&self) -> &ImportCandidate<'db> {
        &self.import_candidate
    }

    pub fn search_for_imports(
        &self,
        sema: &Semantics<'db, RootDatabase>,
        cfg: ImportPathConfig,
        prefix_kind: PrefixKind,
    ) -> impl Iterator<Item = LocatedImport> {
        let _p = tracing::info_span!("ImportAssets::search_for_imports").entered();
        self.search_for(sema, Some(prefix_kind), cfg)
    }

    /// This may return non-absolute paths if a part of the returned path is already imported into scope.
    pub fn search_for_relative_paths(
        &self,
        sema: &Semantics<'db, RootDatabase>,
        cfg: ImportPathConfig,
    ) -> impl Iterator<Item = LocatedImport> {
        let _p = tracing::info_span!("ImportAssets::search_for_relative_paths").entered();
        self.search_for(sema, None, cfg)
    }

    /// Requires imports to by prefix instead of fuzzily.
    pub fn path_fuzzy_name_to_prefix(&mut self) {
        if let ImportCandidate::Path(PathImportCandidate { name: to_import, .. }) =
            &mut self.import_candidate
        {
            let (name, case_sensitive) = match to_import {
                NameToImport::Fuzzy(name, case_sensitive) => {
                    (std::mem::take(name), *case_sensitive)
                }
                _ => return,
            };
            *to_import = NameToImport::Prefix(name, case_sensitive);
        }
    }

    /// Requires imports to match exactly instead of fuzzily.
    pub fn path_fuzzy_name_to_exact(&mut self) {
        if let ImportCandidate::Path(PathImportCandidate { name: to_import, .. }) =
            &mut self.import_candidate
        {
            let (name, case_sensitive) = match to_import {
                NameToImport::Fuzzy(name, case_sensitive) => {
                    (std::mem::take(name), *case_sensitive)
                }
                _ => return,
            };
            *to_import = NameToImport::Exact(name, case_sensitive);
        }
    }

    fn search_for(
        &self,
        sema: &Semantics<'db, RootDatabase>,
        prefixed: Option<PrefixKind>,
        cfg: ImportPathConfig,
    ) -> impl Iterator<Item = LocatedImport> {
        let _p = tracing::info_span!("ImportAssets::search_for").entered();

        let scope = match sema.scope(&self.candidate_node) {
            Some(it) => it,
            None => return <FxIndexSet<_>>::default().into_iter(),
        };
        let db = sema.db;
        let krate = self.module_with_candidate.krate();
        let scope_definitions = self.scope_definitions(sema);
        let mod_path = |item| {
            get_mod_path(
                db,
                item_for_path_search(db, item)?,
                &self.module_with_candidate,
                prefixed,
                cfg,
            )
            .filter(|path| path.len() > 1)
        };

        match &self.import_candidate {
            ImportCandidate::Path(path_candidate) => path_applicable_imports(
                db,
                &scope,
                krate,
                path_candidate,
                mod_path,
                |item_to_import| !scope_definitions.contains(&ScopeDef::from(item_to_import)),
            ),
            ImportCandidate::TraitAssocItem(trait_candidate)
            | ImportCandidate::TraitMethod(trait_candidate) => trait_applicable_items(
                db,
                krate,
                &scope,
                trait_candidate,
                matches!(self.import_candidate, ImportCandidate::TraitAssocItem(_)),
                mod_path,
                |trait_to_import| {
                    !scope_definitions
                        .contains(&ScopeDef::ModuleDef(ModuleDef::Trait(trait_to_import)))
                },
            ),
        }
        .into_iter()
    }

    fn scope_definitions(&self, sema: &Semantics<'_, RootDatabase>) -> FxHashSet<ScopeDef> {
        let _p = tracing::info_span!("ImportAssets::scope_definitions").entered();
        let mut scope_definitions = FxHashSet::default();
        if let Some(scope) = sema.scope(&self.candidate_node) {
            scope.process_all_names(&mut |_, scope_def| {
                scope_definitions.insert(scope_def);
            });
        }
        scope_definitions
    }
}

fn path_applicable_imports(
    db: &RootDatabase,
    scope: &SemanticsScope<'_>,
    current_crate: Crate,
    path_candidate: &PathImportCandidate,
    mod_path: impl Fn(ItemInNs) -> Option<ModPath> + Copy,
    scope_filter: impl Fn(ItemInNs) -> bool + Copy,
) -> FxIndexSet<LocatedImport> {
    let _p = tracing::info_span!("ImportAssets::path_applicable_imports").entered();

    match &*path_candidate.qualifier {
        [] => {
            items_locator::items_with_name(
                db,
                current_crate,
                path_candidate.name.clone(),
                // FIXME: we could look up assoc items by the input and propose those in completion,
                // but that requires more preparation first:
                // * store non-trait assoc items in import_map to fully enable this lookup
                // * ensure that does not degrade the performance (benchmark it)
                // * write more logic to check for corresponding trait presence requirement (we're unable to flyimport multiple item right now)
                // * improve the associated completion item matching and/or scoring to ensure no noisy completions appear
                //
                // see also an ignored test under FIXME comment in the qualify_path.rs module
                AssocSearchMode::Exclude,
            )
            .filter_map(|(item, do_not_complete)| {
                if !scope_filter(item) {
                    return None;
                }
                let mod_path = mod_path(item)?;
                Some(LocatedImport::new(
                    mod_path,
                    item,
                    item,
                    CompleteInFlyimport(do_not_complete != Complete::IgnoreFlyimport),
                ))
            })
            .take(DEFAULT_QUERY_SEARCH_LIMIT)
            .collect()
        }
        // we have some unresolved qualifier that we search an import for
        // The key here is that whatever we import must form a resolved path for the remainder of
        // what follows
        // FIXME: This doesn't handle visibility
        [first_qsegment, qualifier_rest @ ..] => items_locator::items_with_name(
            db,
            current_crate,
            NameToImport::Exact(first_qsegment.as_str().to_owned(), true),
            AssocSearchMode::Exclude,
        )
        .filter_map(|(item, do_not_complete)| {
            // we found imports for `first_qsegment`, now we need to filter these imports by whether
            // they result in resolving the rest of the path successfully
            validate_resolvable(
                db,
                scope,
                mod_path,
                scope_filter,
                &path_candidate.name,
                item,
                qualifier_rest,
                CompleteInFlyimport(do_not_complete != Complete::IgnoreFlyimport),
            )
        })
        .take(DEFAULT_QUERY_SEARCH_LIMIT)
        .collect(),
    }
}

/// Validates and builds an import for `resolved_qualifier` if the `unresolved_qualifier` appended
/// to it resolves and there is a validate `candidate` after that.
fn validate_resolvable(
    db: &RootDatabase,
    scope: &SemanticsScope<'_>,
    mod_path: impl Fn(ItemInNs) -> Option<ModPath>,
    scope_filter: impl Fn(ItemInNs) -> bool,
    candidate: &NameToImport,
    resolved_qualifier: ItemInNs,
    unresolved_qualifier: &[Name],
    complete_in_flyimport: CompleteInFlyimport,
) -> Option<LocatedImport> {
    let _p = tracing::info_span!("ImportAssets::import_for_item").entered();

    let qualifier = {
        let mut adjusted_resolved_qualifier = resolved_qualifier;
        if !unresolved_qualifier.is_empty() {
            match resolved_qualifier {
                ItemInNs::Types(ModuleDef::Module(module)) => {
                    adjusted_resolved_qualifier = module
                        .resolve_mod_path(db, unresolved_qualifier.iter().cloned())?
                        .next()?;
                }
                // can't resolve multiple segments for non-module item path bases
                _ => return None,
            }
        }

        match adjusted_resolved_qualifier {
            ItemInNs::Types(def) => def,
            _ => return None,
        }
    };
    let import_path_candidate = mod_path(resolved_qualifier)?;
    let ty = match qualifier {
        ModuleDef::Module(module) => {
            return items_locator::items_with_name_in_module(
                db,
                module,
                candidate.clone(),
                AssocSearchMode::Exclude,
                |it| match scope_filter(it) {
                    true => ControlFlow::Break(it),
                    false => ControlFlow::Continue(()),
                },
            )
            .map(|item| {
                LocatedImport::new(
                    import_path_candidate,
                    resolved_qualifier,
                    item,
                    complete_in_flyimport,
                )
            });
        }
        // FIXME
        ModuleDef::Trait(_) => return None,
        // FIXME
        ModuleDef::TraitAlias(_) => return None,
        ModuleDef::TypeAlias(alias) => alias.ty(db),
        ModuleDef::BuiltinType(builtin) => builtin.ty(db),
        ModuleDef::Adt(adt) => adt.ty(db),
        _ => return None,
    };
    ty.iterate_path_candidates(db, scope, &FxHashSet::default(), None, None, |assoc| {
        // FIXME: Support extra trait imports
        if assoc.container_or_implemented_trait(db).is_some() {
            return None;
        }
        let name = assoc.name(db)?;
        let is_match = match candidate {
            NameToImport::Prefix(text, true) => name.as_str().starts_with(text),
            NameToImport::Prefix(text, false) => {
                name.as_str().chars().zip(text.chars()).all(|(name_char, candidate_char)| {
                    name_char.eq_ignore_ascii_case(&candidate_char)
                })
            }
            NameToImport::Exact(text, true) => name.as_str() == text,
            NameToImport::Exact(text, false) => name.as_str().eq_ignore_ascii_case(text),
            NameToImport::Fuzzy(text, true) => text.chars().all(|c| name.as_str().contains(c)),
            NameToImport::Fuzzy(text, false) => text
                .chars()
                .all(|c| name.as_str().chars().any(|name_char| name_char.eq_ignore_ascii_case(&c))),
        };
        if !is_match {
            return None;
        }
        Some(LocatedImport::new(
            import_path_candidate.clone(),
            resolved_qualifier,
            assoc_to_item(assoc),
            complete_in_flyimport,
        ))
    })
}

pub fn item_for_path_search(db: &RootDatabase, item: ItemInNs) -> Option<ItemInNs> {
    Some(match item {
        ItemInNs::Types(_) | ItemInNs::Values(_) => match item_as_assoc(db, item) {
            Some(assoc_item) => item_for_path_search_assoc(db, assoc_item)?,
            None => item,
        },
        ItemInNs::Macros(_) => item,
    })
}

fn item_for_path_search_assoc(db: &RootDatabase, assoc_item: AssocItem) -> Option<ItemInNs> {
    Some(match assoc_item.container(db) {
        AssocItemContainer::Trait(trait_) => ItemInNs::from(ModuleDef::from(trait_)),
        AssocItemContainer::Impl(impl_) => {
            ItemInNs::from(ModuleDef::from(impl_.self_ty(db).as_adt()?))
        }
    })
}

fn trait_applicable_items<'db>(
    db: &'db RootDatabase,
    current_crate: Crate,
    scope: &SemanticsScope<'db>,
    trait_candidate: &TraitImportCandidate<'db>,
    trait_assoc_item: bool,
    mod_path: impl Fn(ItemInNs) -> Option<ModPath>,
    scope_filter: impl Fn(hir::Trait) -> bool,
) -> FxIndexSet<LocatedImport> {
    let _p = tracing::info_span!("ImportAssets::trait_applicable_items").entered();

    let inherent_traits = trait_candidate.receiver_ty.applicable_inherent_traits(db);
    let env_traits = trait_candidate.receiver_ty.env_traits(db);
    let related_traits = inherent_traits.chain(env_traits).collect::<FxHashSet<_>>();

    let mut required_assoc_items = FxHashMap::default();
    let mut trait_candidates: FxHashSet<_> = items_locator::items_with_name(
        db,
        current_crate,
        trait_candidate.assoc_item_name.clone(),
        AssocSearchMode::AssocItemsOnly,
    )
    .filter_map(|(input, do_not_complete)| Some((item_as_assoc(db, input)?, do_not_complete)))
    .filter_map(|(assoc, do_not_complete)| {
        if !trait_assoc_item && matches!(assoc, AssocItem::Const(_) | AssocItem::TypeAlias(_)) {
            return None;
        }

        let assoc_item_trait = assoc.container_trait(db)?;
        if related_traits.contains(&assoc_item_trait) {
            return None;
        }
        required_assoc_items
            .insert(assoc, CompleteInFlyimport(do_not_complete != Complete::IgnoreFlyimport));
        Some(assoc_item_trait.into())
    })
    .collect();

    let autoderef_method_receiver = {
        let mut deref_chain = trait_candidate.receiver_ty.autoderef(db).collect::<Vec<_>>();
        // As a last step, we can do array unsizing (that's the only unsizing that rustc does for method receivers!)
        if let Some((ty, _len)) = deref_chain.last().and_then(|ty| ty.as_array(db)) {
            let slice = Type::new_slice(ty);
            deref_chain.push(slice);
        }
        deref_chain
            .into_iter()
            .filter_map(|ty| Some((ty.krate(db).into(), ty.fingerprint_for_trait_impl()?)))
            .sorted()
            .unique()
            .collect::<Vec<_>>()
    };

    // can be empty if the entire deref chain is has no valid trait impl fingerprints
    if autoderef_method_receiver.is_empty() {
        return Default::default();
    }

    // in order to handle implied bounds through an associated type, keep all traits if any
    // type in the deref chain matches `TyFingerprint::Unnameable`. This fingerprint
    // won't be in `TraitImpls` anyways, as `TraitImpls` only contains actual implementations.
    if !autoderef_method_receiver
        .iter()
        .any(|(_, fingerprint)| matches!(fingerprint, TyFingerprint::Unnameable))
    {
        trait_candidates.retain(|&candidate_trait_id| {
            // we care about the following cases:
            // 1. Trait's definition crate
            // 2. Definition crates for all trait's generic arguments
            //     a. This is recursive for fundamental types: `Into<Box<A>> for ()`` is OK, but
            //        `Into<Vec<A>> for ()`` is *not*.
            // 3. Receiver type definition crate
            //    a. This is recursive for fundamental types
            let defining_crate_for_trait = Trait::from(candidate_trait_id).krate(db);

            let trait_impls_in_crate = db.trait_impls_in_crate(defining_crate_for_trait.into());
            let definitions_exist_in_trait_crate =
                autoderef_method_receiver.iter().any(|&(_, fingerprint)| {
                    trait_impls_in_crate
                        .has_impls_for_trait_and_self_ty(candidate_trait_id, fingerprint)
                });
            // this is a closure for laziness: if `definitions_exist_in_trait_crate` is true,
            // we can avoid a second db lookup.
            let definitions_exist_in_receiver_crate = || {
                autoderef_method_receiver.iter().any(|&(krate, fingerprint)| {
                    db.trait_impls_in_crate(krate)
                        .has_impls_for_trait_and_self_ty(candidate_trait_id, fingerprint)
                })
            };

            definitions_exist_in_trait_crate || definitions_exist_in_receiver_crate()
        });
    }

    let mut located_imports = FxIndexSet::default();
    let mut trait_import_paths = FxHashMap::default();

    if trait_assoc_item {
        trait_candidate.receiver_ty.iterate_path_candidates(
            db,
            scope,
            &trait_candidates,
            None,
            None,
            |assoc| {
                if let Some(&complete_in_flyimport) = required_assoc_items.get(&assoc) {
                    let located_trait = assoc.container_trait(db).filter(|&it| scope_filter(it))?;
                    let trait_item = ItemInNs::from(ModuleDef::from(located_trait));
                    let import_path = trait_import_paths
                        .entry(trait_item)
                        .or_insert_with(|| mod_path(trait_item))
                        .clone()?;
                    located_imports.insert(LocatedImport::new(
                        import_path,
                        trait_item,
                        assoc_to_item(assoc),
                        complete_in_flyimport,
                    ));
                }
                None::<()>
            },
        )
    } else {
        trait_candidate.receiver_ty.iterate_method_candidates_with_traits(
            db,
            scope,
            &trait_candidates,
            None,
            None,
            |function| {
                let assoc = function.as_assoc_item(db)?;
                if let Some(&complete_in_flyimport) = required_assoc_items.get(&assoc) {
                    let located_trait = assoc.container_trait(db).filter(|&it| scope_filter(it))?;
                    let trait_item = ItemInNs::from(ModuleDef::from(located_trait));
                    let import_path = trait_import_paths
                        .entry(trait_item)
                        .or_insert_with(|| mod_path(trait_item))
                        .clone()?;
                    located_imports.insert(LocatedImport::new(
                        import_path,
                        trait_item,
                        assoc_to_item(assoc),
                        complete_in_flyimport,
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

#[tracing::instrument(skip_all)]
fn get_mod_path(
    db: &RootDatabase,
    item_to_search: ItemInNs,
    module_with_candidate: &Module,
    prefixed: Option<PrefixKind>,
    cfg: ImportPathConfig,
) -> Option<ModPath> {
    if let Some(prefix_kind) = prefixed {
        module_with_candidate.find_use_path(db, item_to_search, prefix_kind, cfg)
    } else {
        module_with_candidate.find_path(db, item_to_search, cfg)
    }
}

impl<'db> ImportCandidate<'db> {
    fn for_method_call(
        sema: &Semantics<'db, RootDatabase>,
        method_call: &ast::MethodCallExpr,
    ) -> Option<Self> {
        match sema.resolve_method_call(method_call) {
            Some(_) => None,
            None => Some(Self::TraitMethod(TraitImportCandidate {
                receiver_ty: sema.type_of_expr(&method_call.receiver()?)?.adjusted(),
                assoc_item_name: NameToImport::exact_case_sensitive(
                    method_call.name_ref()?.to_string(),
                ),
            })),
        }
    }

    fn for_regular_path(sema: &Semantics<'db, RootDatabase>, path: &ast::Path) -> Option<Self> {
        if sema.resolve_path(path).is_some() {
            return None;
        }
        path_import_candidate(
            sema,
            path.qualifier(),
            NameToImport::exact_case_sensitive(path.segment()?.name_ref()?.to_string()),
        )
    }

    fn for_name(sema: &Semantics<'db, RootDatabase>, name: &ast::Name) -> Option<Self> {
        if sema
            .scope(name.syntax())?
            .speculative_resolve(&make::ext::ident_path(&name.text()))
            .is_some()
        {
            return None;
        }
        Some(ImportCandidate::Path(PathImportCandidate {
            qualifier: vec![],
            name: NameToImport::exact_case_sensitive(name.to_string()),
        }))
    }

    fn for_fuzzy_path(
        qualifier: Option<ast::Path>,
        fuzzy_name: String,
        sema: &Semantics<'db, RootDatabase>,
    ) -> Option<Self> {
        path_import_candidate(sema, qualifier, NameToImport::fuzzy(fuzzy_name))
    }
}

fn path_import_candidate<'db>(
    sema: &Semantics<'db, RootDatabase>,
    qualifier: Option<ast::Path>,
    name: NameToImport,
) -> Option<ImportCandidate<'db>> {
    Some(match qualifier {
        Some(qualifier) => match sema.resolve_path(&qualifier) {
            Some(PathResolution::Def(ModuleDef::BuiltinType(_))) | None => {
                if qualifier.first_qualifier().is_none_or(|it| sema.resolve_path(&it).is_none()) {
                    let qualifier = qualifier
                        .segments()
                        .map(|seg| seg.name_ref().map(|name| Name::new_root(&name.text())))
                        .collect::<Option<Vec<_>>>()?;
                    ImportCandidate::Path(PathImportCandidate { qualifier, name })
                } else {
                    return None;
                }
            }
            Some(PathResolution::Def(ModuleDef::Adt(assoc_item_path))) => {
                ImportCandidate::TraitAssocItem(TraitImportCandidate {
                    receiver_ty: assoc_item_path.ty(sema.db),
                    assoc_item_name: name,
                })
            }
            Some(PathResolution::Def(ModuleDef::TypeAlias(alias))) => {
                let ty = alias.ty(sema.db);
                if ty.as_adt().is_some() {
                    ImportCandidate::TraitAssocItem(TraitImportCandidate {
                        receiver_ty: ty,
                        assoc_item_name: name,
                    })
                } else {
                    return None;
                }
            }
            Some(_) => return None,
        },
        None => ImportCandidate::Path(PathImportCandidate { qualifier: vec![], name }),
    })
}

fn item_as_assoc(db: &RootDatabase, item: ItemInNs) -> Option<AssocItem> {
    item.into_module_def().as_assoc_item(db)
}
