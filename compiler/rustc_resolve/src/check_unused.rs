//
// Unused import checking
//
// Although this is mostly a lint pass, it lives in here because it depends on
// resolve data structures and because it finalises the privacy information for
// `use` items.
//
// Unused trait imports can't be checked until the method resolution. We save
// candidates here, and do the actual check in rustc_hir_analysis/check_unused.rs.
//
// Checking for unused imports is split into three steps:
//
//  - `UnusedImportCheckVisitor` walks the AST to find all the unused imports
//    inside of `UseTree`s, recording their `NodeId`s and grouping them by
//    the parent `use` item
//
//  - `calc_unused_spans` then walks over all the `use` items marked in the
//    previous step to collect the spans associated with the `NodeId`s and to
//    calculate the spans that can be removed by rustfix; This is done in a
//    separate step to be able to collapse the adjacent spans that rustfix
//    will remove
//
//  - `check_unused` finally emits the diagnostics based on the data generated
//    in the last step

use rustc_ast as ast;
use rustc_ast::visit::{self, Visitor};
use rustc_data_structures::fx::{FxHashMap, FxIndexMap, FxIndexSet};
use rustc_data_structures::unord::UnordSet;
use rustc_errors::MultiSpan;
use rustc_hir::def::{DefKind, Res};
use rustc_session::lint::BuiltinLintDiag;
use rustc_session::lint::builtin::{
    MACRO_USE_EXTERN_CRATE, UNUSED_EXTERN_CRATES, UNUSED_IMPORTS, UNUSED_QUALIFICATIONS,
};
use rustc_span::{DUMMY_SP, Ident, Span, kw};

use crate::imports::{Import, ImportKind};
use crate::{LexicalScopeBinding, NameBindingKind, Resolver, module_to_string};

struct UnusedImport {
    use_tree: ast::UseTree,
    use_tree_id: ast::NodeId,
    item_span: Span,
    unused: UnordSet<ast::NodeId>,
}

impl UnusedImport {
    fn add(&mut self, id: ast::NodeId) {
        self.unused.insert(id);
    }
}

struct UnusedImportCheckVisitor<'a, 'ra, 'tcx> {
    r: &'a mut Resolver<'ra, 'tcx>,
    /// All the (so far) unused imports, grouped path list
    unused_imports: FxIndexMap<ast::NodeId, UnusedImport>,
    extern_crate_items: Vec<ExternCrateToLint>,
    base_use_tree: Option<&'a ast::UseTree>,
    base_id: ast::NodeId,
    item_span: Span,
}

struct ExternCrateToLint {
    id: ast::NodeId,
    /// Span from the item
    span: Span,
    /// Span to use to suggest complete removal.
    span_with_attributes: Span,
    /// Span of the visibility, if any.
    vis_span: Span,
    /// Whether the item has attrs.
    has_attrs: bool,
    /// Name used to refer to the crate.
    ident: Ident,
    /// Whether the statement renames the crate `extern crate orig_name as new_name;`.
    renames: bool,
}

impl<'a, 'ra, 'tcx> UnusedImportCheckVisitor<'a, 'ra, 'tcx> {
    // We have information about whether `use` (import) items are actually
    // used now. If an import is not used at all, we signal a lint error.
    fn check_import(&mut self, id: ast::NodeId) {
        let used = self.r.used_imports.contains(&id);
        let def_id = self.r.local_def_id(id);
        if !used {
            if self.r.maybe_unused_trait_imports.contains(&def_id) {
                // Check later.
                return;
            }
            self.unused_import(self.base_id).add(id);
        } else {
            // This trait import is definitely used, in a way other than
            // method resolution.
            // FIXME(#120456) - is `swap_remove` correct?
            self.r.maybe_unused_trait_imports.swap_remove(&def_id);
            if let Some(i) = self.unused_imports.get_mut(&self.base_id) {
                i.unused.remove(&id);
            }
        }
    }

    fn check_use_tree(&mut self, use_tree: &'a ast::UseTree, id: ast::NodeId) {
        if self.r.effective_visibilities.is_exported(self.r.local_def_id(id)) {
            self.check_import_as_underscore(use_tree, id);
            return;
        }

        if let ast::UseTreeKind::Nested { ref items, .. } = use_tree.kind {
            if items.is_empty() {
                self.unused_import(self.base_id).add(id);
            }
        } else {
            self.check_import(id);
        }
    }

    fn unused_import(&mut self, id: ast::NodeId) -> &mut UnusedImport {
        let use_tree_id = self.base_id;
        let use_tree = self.base_use_tree.unwrap().clone();
        let item_span = self.item_span;

        self.unused_imports.entry(id).or_insert_with(|| UnusedImport {
            use_tree,
            use_tree_id,
            item_span,
            unused: Default::default(),
        })
    }

    fn check_import_as_underscore(&mut self, item: &ast::UseTree, id: ast::NodeId) {
        match item.kind {
            ast::UseTreeKind::Simple(Some(ident)) => {
                if ident.name == kw::Underscore
                    && !self.r.import_res_map.get(&id).is_some_and(|per_ns| {
                        matches!(
                            per_ns.type_ns,
                            Some(Res::Def(DefKind::Trait | DefKind::TraitAlias, _))
                        )
                    })
                {
                    self.unused_import(self.base_id).add(id);
                }
            }
            ast::UseTreeKind::Nested { ref items, .. } => self.check_imports_as_underscore(items),
            _ => {}
        }
    }

    fn check_imports_as_underscore(&mut self, items: &[(ast::UseTree, ast::NodeId)]) {
        for (item, id) in items {
            self.check_import_as_underscore(item, *id);
        }
    }

    fn report_unused_extern_crate_items(
        &mut self,
        maybe_unused_extern_crates: FxHashMap<ast::NodeId, Span>,
    ) {
        let tcx = self.r.tcx();
        for extern_crate in &self.extern_crate_items {
            let warn_if_unused = !extern_crate.ident.name.as_str().starts_with('_');

            // If the crate is fully unused, we suggest removing it altogether.
            // We do this in any edition.
            if warn_if_unused {
                if let Some(&span) = maybe_unused_extern_crates.get(&extern_crate.id) {
                    self.r.lint_buffer.buffer_lint(
                        UNUSED_EXTERN_CRATES,
                        extern_crate.id,
                        span,
                        BuiltinLintDiag::UnusedExternCrate {
                            span: extern_crate.span,
                            removal_span: extern_crate.span_with_attributes,
                        },
                    );
                    continue;
                }
            }

            // If we are not in Rust 2018 edition, then we don't make any further
            // suggestions.
            if !tcx.sess.at_least_rust_2018() {
                continue;
            }

            // If the extern crate has any attributes, they may have funky
            // semantics we can't faithfully represent using `use` (most
            // notably `#[macro_use]`). Ignore it.
            if extern_crate.has_attrs {
                continue;
            }

            // If the extern crate is renamed, then we cannot suggest replacing it with a use as this
            // would not insert the new name into the prelude, where other imports in the crate may be
            // expecting it.
            if extern_crate.renames {
                continue;
            }

            // If the extern crate isn't in the extern prelude,
            // there is no way it can be written as a `use`.
            if self
                .r
                .extern_prelude
                .get(&extern_crate.ident)
                .is_none_or(|entry| entry.introduced_by_item)
            {
                continue;
            }

            let module = self
                .r
                .get_nearest_non_block_module(self.r.local_def_id(extern_crate.id).to_def_id());
            if module.no_implicit_prelude {
                // If the module has `no_implicit_prelude`, then we don't suggest
                // replacing the extern crate with a use, as it would not be
                // inserted into the prelude. User writes `extern` style deliberately.
                continue;
            }

            let vis_span = extern_crate
                .vis_span
                .find_ancestor_inside(extern_crate.span)
                .unwrap_or(extern_crate.vis_span);
            let ident_span = extern_crate
                .ident
                .span
                .find_ancestor_inside(extern_crate.span)
                .unwrap_or(extern_crate.ident.span);
            self.r.lint_buffer.buffer_lint(
                UNUSED_EXTERN_CRATES,
                extern_crate.id,
                extern_crate.span,
                BuiltinLintDiag::ExternCrateNotIdiomatic { vis_span, ident_span },
            );
        }
    }
}

impl<'a, 'ra, 'tcx> Visitor<'a> for UnusedImportCheckVisitor<'a, 'ra, 'tcx> {
    fn visit_item(&mut self, item: &'a ast::Item) {
        self.item_span = item.span_with_attributes();
        match &item.kind {
            // Ignore is_public import statements because there's no way to be sure
            // whether they're used or not. Also ignore imports with a dummy span
            // because this means that they were generated in some fashion by the
            // compiler and we don't need to consider them.
            ast::ItemKind::Use(..) if item.span.is_dummy() => return,
            // Use the base UseTree's NodeId as the item id
            // This allows the grouping of all the lints in the same item
            ast::ItemKind::Use(use_tree) => {
                self.base_id = item.id;
                self.base_use_tree = Some(use_tree);
                self.check_use_tree(use_tree, item.id);
            }
            &ast::ItemKind::ExternCrate(orig_name, ident) => {
                self.extern_crate_items.push(ExternCrateToLint {
                    id: item.id,
                    span: item.span,
                    vis_span: item.vis.span,
                    span_with_attributes: item.span_with_attributes(),
                    has_attrs: !item.attrs.is_empty(),
                    ident,
                    renames: orig_name.is_some(),
                });
            }
            _ => {}
        }

        visit::walk_item(self, item);
    }

    fn visit_nested_use_tree(&mut self, use_tree: &'a ast::UseTree, id: ast::NodeId) {
        self.check_use_tree(use_tree, id);
        visit::walk_use_tree(self, use_tree);
    }
}

enum UnusedSpanResult {
    Used,
    Unused { spans: Vec<Span>, remove: Span },
    PartialUnused { spans: Vec<Span>, remove: Vec<Span> },
}

fn calc_unused_spans(
    unused_import: &UnusedImport,
    use_tree: &ast::UseTree,
    use_tree_id: ast::NodeId,
) -> UnusedSpanResult {
    // The full span is the whole item's span if this current tree is not nested inside another
    // This tells rustfix to remove the whole item if all the imports are unused
    let full_span = if unused_import.use_tree.span == use_tree.span {
        unused_import.item_span
    } else {
        use_tree.span
    };
    match use_tree.kind {
        ast::UseTreeKind::Simple(..) | ast::UseTreeKind::Glob => {
            if unused_import.unused.contains(&use_tree_id) {
                UnusedSpanResult::Unused { spans: vec![use_tree.span], remove: full_span }
            } else {
                UnusedSpanResult::Used
            }
        }
        ast::UseTreeKind::Nested { items: ref nested, span: tree_span } => {
            if nested.is_empty() {
                return UnusedSpanResult::Unused { spans: vec![use_tree.span], remove: full_span };
            }

            let mut unused_spans = Vec::new();
            let mut to_remove = Vec::new();
            let mut used_children = 0;
            let mut contains_self = false;
            let mut previous_unused = false;
            for (pos, (use_tree, use_tree_id)) in nested.iter().enumerate() {
                let remove = match calc_unused_spans(unused_import, use_tree, *use_tree_id) {
                    UnusedSpanResult::Used => {
                        used_children += 1;
                        None
                    }
                    UnusedSpanResult::Unused { mut spans, remove } => {
                        unused_spans.append(&mut spans);
                        Some(remove)
                    }
                    UnusedSpanResult::PartialUnused { mut spans, remove: mut to_remove_extra } => {
                        used_children += 1;
                        unused_spans.append(&mut spans);
                        to_remove.append(&mut to_remove_extra);
                        None
                    }
                };
                if let Some(remove) = remove {
                    let remove_span = if nested.len() == 1 {
                        remove
                    } else if pos == nested.len() - 1 || used_children > 0 {
                        // Delete everything from the end of the last import, to delete the
                        // previous comma
                        nested[pos - 1].0.span.shrink_to_hi().to(use_tree.span)
                    } else {
                        // Delete everything until the next import, to delete the trailing commas
                        use_tree.span.to(nested[pos + 1].0.span.shrink_to_lo())
                    };

                    // Try to collapse adjacent spans into a single one. This prevents all cases of
                    // overlapping removals, which are not supported by rustfix
                    if previous_unused && !to_remove.is_empty() {
                        let previous = to_remove.pop().unwrap();
                        to_remove.push(previous.to(remove_span));
                    } else {
                        to_remove.push(remove_span);
                    }
                }
                contains_self |= use_tree.prefix == kw::SelfLower
                    && matches!(use_tree.kind, ast::UseTreeKind::Simple(_))
                    && !unused_import.unused.contains(&use_tree_id);
                previous_unused = remove.is_some();
            }
            if unused_spans.is_empty() {
                UnusedSpanResult::Used
            } else if used_children == 0 {
                UnusedSpanResult::Unused { spans: unused_spans, remove: full_span }
            } else {
                // If there is only one remaining child that is used, the braces around the use
                // tree are not needed anymore. In that case, we determine the span of the left
                // brace and the right brace, and tell rustfix to remove them as well.
                //
                // This means that `use a::{B, C};` will be turned into `use a::B;` rather than
                // `use a::{B};`, removing a rustfmt roundtrip.
                //
                // Note that we cannot remove the braces if the only item inside the use tree is
                // `self`: `use foo::{self};` is valid Rust syntax, while `use foo::self;` errors
                // out. We also cannot turn `use foo::{self}` into `use foo`, as the former doesn't
                // import types with the same name as the module.
                if used_children == 1 && !contains_self {
                    // Left brace, from the start of the nested group to the first item.
                    to_remove.push(
                        tree_span.shrink_to_lo().to(nested.first().unwrap().0.span.shrink_to_lo()),
                    );
                    // Right brace, from the end of the last item to the end of the nested group.
                    to_remove.push(
                        nested.last().unwrap().0.span.shrink_to_hi().to(tree_span.shrink_to_hi()),
                    );
                }

                UnusedSpanResult::PartialUnused { spans: unused_spans, remove: to_remove }
            }
        }
    }
}

impl Resolver<'_, '_> {
    pub(crate) fn check_unused(&mut self, krate: &ast::Crate) {
        let tcx = self.tcx;
        let mut maybe_unused_extern_crates = FxHashMap::default();

        for import in self.potentially_unused_imports.iter() {
            match import.kind {
                _ if import.vis.is_public()
                    || import.span.is_dummy()
                    || self.import_use_map.contains_key(import) =>
                {
                    if let ImportKind::MacroUse { .. } = import.kind {
                        if !import.span.is_dummy() {
                            self.lint_buffer.buffer_lint(
                                MACRO_USE_EXTERN_CRATE,
                                import.root_id,
                                import.span,
                                BuiltinLintDiag::MacroUseDeprecated,
                            );
                        }
                    }
                }
                ImportKind::ExternCrate { id, .. } => {
                    let def_id = self.local_def_id(id);
                    if self.extern_crate_map.get(&def_id).is_none_or(|&cnum| {
                        !tcx.is_compiler_builtins(cnum)
                            && !tcx.is_panic_runtime(cnum)
                            && !tcx.has_global_allocator(cnum)
                            && !tcx.has_panic_handler(cnum)
                    }) {
                        maybe_unused_extern_crates.insert(id, import.span);
                    }
                }
                ImportKind::MacroUse { .. } => {
                    self.lint_buffer.buffer_lint(
                        UNUSED_IMPORTS,
                        import.root_id,
                        import.span,
                        BuiltinLintDiag::UnusedMacroUse,
                    );
                }
                _ => {}
            }
        }

        let mut visitor = UnusedImportCheckVisitor {
            r: self,
            unused_imports: Default::default(),
            extern_crate_items: Default::default(),
            base_use_tree: None,
            base_id: ast::DUMMY_NODE_ID,
            item_span: DUMMY_SP,
        };
        visit::walk_crate(&mut visitor, krate);

        visitor.report_unused_extern_crate_items(maybe_unused_extern_crates);

        for unused in visitor.unused_imports.values() {
            let (spans, remove_spans) =
                match calc_unused_spans(unused, &unused.use_tree, unused.use_tree_id) {
                    UnusedSpanResult::Used => continue,
                    UnusedSpanResult::Unused { spans, remove } => (spans, vec![remove]),
                    UnusedSpanResult::PartialUnused { spans, remove } => (spans, remove),
                };

            let ms = MultiSpan::from_spans(spans);

            let mut span_snippets = ms
                .primary_spans()
                .iter()
                .filter_map(|span| tcx.sess.source_map().span_to_snippet(*span).ok())
                .map(|s| format!("`{s}`"))
                .collect::<Vec<String>>();
            span_snippets.sort();

            let remove_whole_use = remove_spans.len() == 1 && remove_spans[0] == unused.item_span;
            let num_to_remove = ms.primary_spans().len();

            // If we are in the `--test` mode, suppress a help that adds the `#[cfg(test)]`
            // attribute; however, if not, suggest adding the attribute. There is no way to
            // retrieve attributes here because we do not have a `TyCtxt` yet.
            let test_module_span = if tcx.sess.is_test_crate() {
                None
            } else {
                let parent_module = visitor.r.get_nearest_non_block_module(
                    visitor.r.local_def_id(unused.use_tree_id).to_def_id(),
                );
                match module_to_string(parent_module) {
                    Some(module)
                        if module == "test"
                            || module == "tests"
                            || module.starts_with("test_")
                            || module.starts_with("tests_")
                            || module.ends_with("_test")
                            || module.ends_with("_tests") =>
                    {
                        Some(parent_module.span)
                    }
                    _ => None,
                }
            };

            visitor.r.lint_buffer.buffer_lint(
                UNUSED_IMPORTS,
                unused.use_tree_id,
                ms,
                BuiltinLintDiag::UnusedImports {
                    remove_whole_use,
                    num_to_remove,
                    remove_spans,
                    test_module_span,
                    span_snippets,
                },
            );
        }

        let unused_imports = visitor.unused_imports;
        let mut check_redundant_imports = FxIndexSet::default();
        for module in self.arenas.local_modules().iter() {
            for (_key, resolution) in self.resolutions(*module).borrow().iter() {
                let resolution = resolution.borrow();

                if let Some(binding) = resolution.binding
                    && let NameBindingKind::Import { import, .. } = binding.kind
                    && let ImportKind::Single { id, .. } = import.kind
                {
                    if let Some(unused_import) = unused_imports.get(&import.root_id)
                        && unused_import.unused.contains(&id)
                    {
                        continue;
                    }

                    check_redundant_imports.insert(import);
                }
            }
        }

        let mut redundant_imports = UnordSet::default();
        for import in check_redundant_imports {
            if self.check_for_redundant_imports(import)
                && let Some(id) = import.id()
            {
                redundant_imports.insert(id);
            }
        }

        // The lint fixes for unused_import and unnecessary_qualification may conflict.
        // Deleting both unused imports and unnecessary segments of an item may result
        // in the item not being found.
        for unn_qua in &self.potentially_unnecessary_qualifications {
            if let LexicalScopeBinding::Item(name_binding) = unn_qua.binding
                && let NameBindingKind::Import { import, .. } = name_binding.kind
                && (is_unused_import(import, &unused_imports)
                    || is_redundant_import(import, &redundant_imports))
            {
                continue;
            }

            self.lint_buffer.buffer_lint(
                UNUSED_QUALIFICATIONS,
                unn_qua.node_id,
                unn_qua.path_span,
                BuiltinLintDiag::UnusedQualifications { removal_span: unn_qua.removal_span },
            );
        }

        fn is_redundant_import(
            import: Import<'_>,
            redundant_imports: &UnordSet<ast::NodeId>,
        ) -> bool {
            if let Some(id) = import.id()
                && redundant_imports.contains(&id)
            {
                return true;
            }
            false
        }

        fn is_unused_import(
            import: Import<'_>,
            unused_imports: &FxIndexMap<ast::NodeId, UnusedImport>,
        ) -> bool {
            if let Some(unused_import) = unused_imports.get(&import.root_id)
                && let Some(id) = import.id()
                && unused_import.unused.contains(&id)
            {
                return true;
            }
            false
        }
    }
}
