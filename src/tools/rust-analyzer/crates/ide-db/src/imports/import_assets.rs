//! Look up accessible paths for items.

use hir::{
    db::HirDatabase, AsAssocItem, AssocItem, AssocItemContainer, Crate, HasCrate, ItemInNs,
    ModPath, Module, ModuleDef, Name, PathResolution, PrefixKind, ScopeDef, Semantics,
    SemanticsScope, Trait, Type,
};
use itertools::{EitherOrBoth, Itertools};
use rustc_hash::{FxHashMap, FxHashSet};
use syntax::{
    ast::{self, make, HasName},
    AstNode, SmolStr, SyntaxNode,
};

use crate::{
    helpers::item_name,
    items_locator::{self, AssocSearchMode, DEFAULT_QUERY_SEARCH_LIMIT},
    RootDatabase,
};

/// A candidate for import, derived during various IDE activities:
/// * completion with imports on the fly proposals
/// * completion edit resolve requests
/// * assists
/// * etc.
#[derive(Debug)]
pub enum ImportCandidate {
    /// A path, qualified (`std::collections::HashMap`) or not (`HashMap`).
    Path(PathImportCandidate),
    /// A trait associated function (with no self parameter) or an associated constant.
    /// For 'test_mod::TestEnum::test_function', `ty` is the `test_mod::TestEnum` expression type
    /// and `name` is the `test_function`
    TraitAssocItem(TraitImportCandidate),
    /// A trait method with self parameter.
    /// For 'test_enum.test_method()', `ty` is the `test_enum` expression type
    /// and `name` is the `test_method`
    TraitMethod(TraitImportCandidate),
}

/// A trait import needed for a given associated item access.
/// For `some::path::SomeStruct::ASSOC_`, contains the
/// type of `some::path::SomeStruct` and `ASSOC_` as the item name.
#[derive(Debug)]
pub struct TraitImportCandidate {
    /// A type of the item that has the associated item accessed at.
    pub receiver_ty: Type,
    /// The associated item name that the trait to import should contain.
    pub assoc_item_name: NameToImport,
}

/// Path import for a given name, qualified or not.
#[derive(Debug)]
pub struct PathImportCandidate {
    /// Optional qualifier before name.
    pub qualifier: Option<Vec<SmolStr>>,
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
        NameToImport::Exact(s, true)
    }

    pub fn fuzzy(s: String) -> NameToImport {
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
pub struct ImportAssets {
    import_candidate: ImportCandidate,
    candidate_node: SyntaxNode,
    module_with_candidate: Module,
}

impl ImportAssets {
    pub fn for_method_call(
        method_call: &ast::MethodCallExpr,
        sema: &Semantics<'_, RootDatabase>,
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
        sema: &Semantics<'_, RootDatabase>,
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

    pub fn for_ident_pat(sema: &Semantics<'_, RootDatabase>, pat: &ast::IdentPat) -> Option<Self> {
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
        sema: &Semantics<'_, RootDatabase>,
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
        receiver_ty: Type,
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
}

impl LocatedImport {
    pub fn new(import_path: ModPath, item_to_import: ItemInNs, original_item: ItemInNs) -> Self {
        Self { import_path, item_to_import, original_item }
    }
}

impl ImportAssets {
    pub fn import_candidate(&self) -> &ImportCandidate {
        &self.import_candidate
    }

    pub fn search_for_imports(
        &self,
        sema: &Semantics<'_, RootDatabase>,
        prefix_kind: PrefixKind,
        prefer_no_std: bool,
        prefer_prelude: bool,
    ) -> impl Iterator<Item = LocatedImport> {
        let _p =
            tracing::span!(tracing::Level::INFO, "import_assets::search_for_imports").entered();
        self.search_for(sema, Some(prefix_kind), prefer_no_std, prefer_prelude)
    }

    /// This may return non-absolute paths if a part of the returned path is already imported into scope.
    pub fn search_for_relative_paths(
        &self,
        sema: &Semantics<'_, RootDatabase>,
        prefer_no_std: bool,
        prefer_prelude: bool,
    ) -> impl Iterator<Item = LocatedImport> {
        let _p = tracing::span!(tracing::Level::INFO, "import_assets::search_for_relative_paths")
            .entered();
        self.search_for(sema, None, prefer_no_std, prefer_prelude)
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
        sema: &Semantics<'_, RootDatabase>,
        prefixed: Option<PrefixKind>,
        prefer_no_std: bool,
        prefer_prelude: bool,
    ) -> impl Iterator<Item = LocatedImport> {
        let _p = tracing::span!(tracing::Level::INFO, "import_assets::search_for").entered();

        let scope = match sema.scope(&self.candidate_node) {
            Some(it) => it,
            None => return <FxHashSet<_>>::default().into_iter(),
        };

        let krate = self.module_with_candidate.krate();
        let scope_definitions = self.scope_definitions(sema);
        let mod_path = |item| {
            get_mod_path(
                sema.db,
                item_for_path_search(sema.db, item)?,
                &self.module_with_candidate,
                prefixed,
                prefer_no_std,
                prefer_prelude,
            )
            .filter(|path| path.len() > 1)
        };

        match &self.import_candidate {
            ImportCandidate::Path(path_candidate) => {
                path_applicable_imports(sema, krate, path_candidate, mod_path, |item_to_import| {
                    !scope_definitions.contains(&ScopeDef::from(item_to_import))
                })
            }
            ImportCandidate::TraitAssocItem(trait_candidate)
            | ImportCandidate::TraitMethod(trait_candidate) => trait_applicable_items(
                sema,
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
        let _p = tracing::span!(tracing::Level::INFO, "import_assets::scope_definitions").entered();
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
    sema: &Semantics<'_, RootDatabase>,
    current_crate: Crate,
    path_candidate: &PathImportCandidate,
    mod_path: impl Fn(ItemInNs) -> Option<ModPath> + Copy,
    scope_filter: impl Fn(ItemInNs) -> bool + Copy,
) -> FxHashSet<LocatedImport> {
    let _p =
        tracing::span!(tracing::Level::INFO, "import_assets::path_applicable_imports").entered();

    match &path_candidate.qualifier {
        None => {
            items_locator::items_with_name(
                sema,
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
            .filter_map(|item| {
                if !scope_filter(item) {
                    return None;
                }
                let mod_path = mod_path(item)?;
                Some(LocatedImport::new(mod_path, item, item))
            })
            .take(DEFAULT_QUERY_SEARCH_LIMIT.inner())
            .collect()
        }
        Some(qualifier) => items_locator::items_with_name(
            sema,
            current_crate,
            path_candidate.name.clone(),
            AssocSearchMode::Include,
        )
        .filter_map(|item| import_for_item(sema.db, mod_path, qualifier, item, scope_filter))
        .take(DEFAULT_QUERY_SEARCH_LIMIT.inner())
        .collect(),
    }
}

fn import_for_item(
    db: &RootDatabase,
    mod_path: impl Fn(ItemInNs) -> Option<ModPath>,
    unresolved_qualifier: &[SmolStr],
    original_item: ItemInNs,
    scope_filter: impl Fn(ItemInNs) -> bool,
) -> Option<LocatedImport> {
    let _p = tracing::span!(tracing::Level::INFO, "import_assets::import_for_item").entered();
    let [first_segment, ..] = unresolved_qualifier else { return None };

    let item_as_assoc = item_as_assoc(db, original_item);

    let (original_item_candidate, trait_item_to_import) = match item_as_assoc {
        Some(assoc_item) => match assoc_item.container(db) {
            AssocItemContainer::Trait(trait_) => {
                let trait_ = ItemInNs::from(ModuleDef::from(trait_));
                (trait_, Some(trait_))
            }
            AssocItemContainer::Impl(impl_) => {
                (ItemInNs::from(ModuleDef::from(impl_.self_ty(db).as_adt()?)), None)
            }
        },
        None => (original_item, None),
    };
    let import_path_candidate = mod_path(original_item_candidate)?;

    let mut import_path_candidate_segments = import_path_candidate.segments().iter().rev();
    let predicate = |it: EitherOrBoth<&SmolStr, &Name>| match it {
        // segments match, check next one
        EitherOrBoth::Both(a, b) if b.as_str() == Some(&**a) => None,
        // segments mismatch / qualifier is longer than the path, bail out
        EitherOrBoth::Both(..) | EitherOrBoth::Left(_) => Some(false),
        // all segments match and we have exhausted the qualifier, proceed
        EitherOrBoth::Right(_) => Some(true),
    };
    if item_as_assoc.is_none() {
        let item_name = item_name(db, original_item)?.as_text()?;
        let last_segment = import_path_candidate_segments.next()?;
        if last_segment.as_str() != Some(&*item_name) {
            return None;
        }
    }
    let ends_with = unresolved_qualifier
        .iter()
        .rev()
        .zip_longest(import_path_candidate_segments)
        .find_map(predicate)
        .unwrap_or(true);
    if !ends_with {
        return None;
    }

    let segment_import = find_import_for_segment(db, original_item_candidate, first_segment)?;

    Some(match (segment_import == original_item_candidate, trait_item_to_import) {
        (true, Some(_)) => {
            // FIXME we should be able to import both the trait and the segment,
            // but it's unclear what to do with overlapping edits (merge imports?)
            // especially in case of lazy completion edit resolutions.
            return None;
        }
        (false, Some(trait_to_import)) if scope_filter(trait_to_import) => {
            LocatedImport::new(mod_path(trait_to_import)?, trait_to_import, original_item)
        }
        (true, None) if scope_filter(original_item_candidate) => {
            LocatedImport::new(import_path_candidate, original_item_candidate, original_item)
        }
        (false, None) if scope_filter(segment_import) => {
            LocatedImport::new(mod_path(segment_import)?, segment_import, original_item)
        }
        _ => return None,
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

fn find_import_for_segment(
    db: &RootDatabase,
    original_item: ItemInNs,
    unresolved_first_segment: &str,
) -> Option<ItemInNs> {
    let segment_is_name = item_name(db, original_item)
        .map(|name| name.to_smol_str() == unresolved_first_segment)
        .unwrap_or(false);

    Some(if segment_is_name {
        original_item
    } else {
        let matching_module =
            module_with_segment_name(db, unresolved_first_segment, original_item)?;
        ItemInNs::from(ModuleDef::from(matching_module))
    })
}

fn module_with_segment_name(
    db: &RootDatabase,
    segment_name: &str,
    candidate: ItemInNs,
) -> Option<Module> {
    let mut current_module = match candidate {
        ItemInNs::Types(module_def_id) => module_def_id.module(db),
        ItemInNs::Values(module_def_id) => module_def_id.module(db),
        ItemInNs::Macros(macro_def_id) => ModuleDef::from(macro_def_id).module(db),
    };
    while let Some(module) = current_module {
        if let Some(module_name) = module.name(db) {
            if module_name.to_smol_str() == segment_name {
                return Some(module);
            }
        }
        current_module = module.parent(db);
    }
    None
}

fn trait_applicable_items(
    sema: &Semantics<'_, RootDatabase>,
    current_crate: Crate,
    scope: &SemanticsScope<'_>,
    trait_candidate: &TraitImportCandidate,
    trait_assoc_item: bool,
    mod_path: impl Fn(ItemInNs) -> Option<ModPath>,
    scope_filter: impl Fn(hir::Trait) -> bool,
) -> FxHashSet<LocatedImport> {
    let _p =
        tracing::span!(tracing::Level::INFO, "import_assets::trait_applicable_items").entered();

    let db = sema.db;

    let inherent_traits = trait_candidate.receiver_ty.applicable_inherent_traits(db);
    let env_traits = trait_candidate.receiver_ty.env_traits(db);
    let related_traits = inherent_traits.chain(env_traits).collect::<FxHashSet<_>>();

    let mut required_assoc_items = FxHashSet::default();
    let mut trait_candidates: FxHashSet<_> = items_locator::items_with_name(
        sema,
        current_crate,
        trait_candidate.assoc_item_name.clone(),
        AssocSearchMode::AssocItemsOnly,
    )
    .filter_map(|input| item_as_assoc(db, input))
    .filter_map(|assoc| {
        if !trait_assoc_item && matches!(assoc, AssocItem::Const(_) | AssocItem::TypeAlias(_)) {
            return None;
        }

        let assoc_item_trait = assoc.container_trait(db)?;
        if related_traits.contains(&assoc_item_trait) {
            return None;
        }
        required_assoc_items.insert(assoc);
        Some(assoc_item_trait.into())
    })
    .collect();

    trait_candidates.retain(|&candidate_trait_id| {
        // we care about the following cases:
        // 1. Trait's definition crate
        // 2. Definition crates for all trait's generic arguments
        //     a. This is recursive for fundamental types: `Into<Box<A>> for ()`` is OK, but
        //        `Into<Vec<A>> for ()`` is *not*.
        // 3. Receiver type definition crate
        //    a. This is recursive for fundamental types
        let defining_crate_for_trait = Trait::from(candidate_trait_id).krate(db);
        let Some(receiver) = trait_candidate.receiver_ty.fingerprint_for_trait_impl() else {
            return false;
        };
        let definitions_exist_in_trait_crate = db
            .trait_impls_in_crate(defining_crate_for_trait.into())
            .has_impls_for_trait_and_self_ty(candidate_trait_id, receiver);

        // this is a closure for laziness: if `definitions_exist_in_trait_crate` is true,
        // we can avoid a second db lookup.
        let definitions_exist_in_receiver_crate = || {
            db.trait_impls_in_crate(trait_candidate.receiver_ty.krate(db).into())
                .has_impls_for_trait_and_self_ty(candidate_trait_id, receiver)
        };

        definitions_exist_in_trait_crate || definitions_exist_in_receiver_crate()
    });

    let mut located_imports = FxHashSet::default();
    let mut trait_import_paths = FxHashMap::default();

    if trait_assoc_item {
        trait_candidate.receiver_ty.iterate_path_candidates(
            db,
            scope,
            &trait_candidates,
            None,
            None,
            |assoc| {
                if required_assoc_items.contains(&assoc) {
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
                if required_assoc_items.contains(&assoc) {
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
    prefer_no_std: bool,
    prefer_prelude: bool,
) -> Option<ModPath> {
    if let Some(prefix_kind) = prefixed {
        module_with_candidate.find_use_path_prefixed(
            db,
            item_to_search,
            prefix_kind,
            prefer_no_std,
            prefer_prelude,
        )
    } else {
        module_with_candidate.find_use_path(db, item_to_search, prefer_no_std, prefer_prelude)
    }
}

impl ImportCandidate {
    fn for_method_call(
        sema: &Semantics<'_, RootDatabase>,
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

    fn for_regular_path(sema: &Semantics<'_, RootDatabase>, path: &ast::Path) -> Option<Self> {
        if sema.resolve_path(path).is_some() {
            return None;
        }
        path_import_candidate(
            sema,
            path.qualifier(),
            NameToImport::exact_case_sensitive(path.segment()?.name_ref()?.to_string()),
        )
    }

    fn for_name(sema: &Semantics<'_, RootDatabase>, name: &ast::Name) -> Option<Self> {
        if sema
            .scope(name.syntax())?
            .speculative_resolve(&make::ext::ident_path(&name.text()))
            .is_some()
        {
            return None;
        }
        Some(ImportCandidate::Path(PathImportCandidate {
            qualifier: None,
            name: NameToImport::exact_case_sensitive(name.to_string()),
        }))
    }

    fn for_fuzzy_path(
        qualifier: Option<ast::Path>,
        fuzzy_name: String,
        sema: &Semantics<'_, RootDatabase>,
    ) -> Option<Self> {
        path_import_candidate(sema, qualifier, NameToImport::fuzzy(fuzzy_name))
    }
}

fn path_import_candidate(
    sema: &Semantics<'_, RootDatabase>,
    qualifier: Option<ast::Path>,
    name: NameToImport,
) -> Option<ImportCandidate> {
    Some(match qualifier {
        Some(qualifier) => match sema.resolve_path(&qualifier) {
            None => {
                if qualifier.first_qualifier().map_or(true, |it| sema.resolve_path(&it).is_none()) {
                    let qualifier = qualifier
                        .segments()
                        .map(|seg| seg.name_ref().map(|name| SmolStr::new(name.text())))
                        .collect::<Option<Vec<_>>>()?;
                    ImportCandidate::Path(PathImportCandidate { qualifier: Some(qualifier), name })
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
        None => ImportCandidate::Path(PathImportCandidate { qualifier: None, name }),
    })
}

fn item_as_assoc(db: &RootDatabase, item: ItemInNs) -> Option<AssocItem> {
    item.as_module_def().and_then(|module_def| module_def.as_assoc_item(db))
}
