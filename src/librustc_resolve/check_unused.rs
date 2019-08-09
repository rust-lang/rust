//
// Unused import checking
//
// Although this is mostly a lint pass, it lives in here because it depends on
// resolve data structures and because it finalises the privacy information for
// `use` directives.
//
// Unused trait imports can't be checked until the method resolution. We save
// candidates here, and do the actual check in librustc_typeck/check_unused.rs.
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
//  - `check_crate` finally emits the diagnostics based on the data generated
//    in the last step

use std::ops::{Deref, DerefMut};

use crate::Resolver;
use crate::resolve_imports::ImportDirectiveSubclass;

use rustc::util::nodemap::NodeMap;
use rustc::{lint, ty};
use rustc_data_structures::fx::FxHashSet;
use syntax::ast;
use syntax::visit::{self, Visitor};
use syntax_pos::{Span, MultiSpan, DUMMY_SP};

struct UnusedImport<'a> {
    use_tree: &'a ast::UseTree,
    use_tree_id: ast::NodeId,
    item_span: Span,
    unused: FxHashSet<ast::NodeId>,
}

impl<'a> UnusedImport<'a> {
    fn add(&mut self, id: ast::NodeId) {
        self.unused.insert(id);
    }
}

struct UnusedImportCheckVisitor<'a, 'b> {
    resolver: &'a mut Resolver<'b>,
    /// All the (so far) unused imports, grouped path list
    unused_imports: NodeMap<UnusedImport<'a>>,
    base_use_tree: Option<&'a ast::UseTree>,
    base_id: ast::NodeId,
    item_span: Span,
}

// Deref and DerefMut impls allow treating UnusedImportCheckVisitor as Resolver.
impl<'a, 'b> Deref for UnusedImportCheckVisitor<'a, 'b> {
    type Target = Resolver<'b>;

    fn deref<'c>(&'c self) -> &'c Resolver<'b> {
        &*self.resolver
    }
}

impl<'a, 'b> DerefMut for UnusedImportCheckVisitor<'a, 'b> {
    fn deref_mut<'c>(&'c mut self) -> &'c mut Resolver<'b> {
        &mut *self.resolver
    }
}

impl<'a, 'b> UnusedImportCheckVisitor<'a, 'b> {
    // We have information about whether `use` (import) directives are actually
    // used now. If an import is not used at all, we signal a lint error.
    fn check_import(&mut self, id: ast::NodeId) {
        let mut used = false;
        self.per_ns(|this, ns| used |= this.used_imports.contains(&(id, ns)));
        if !used {
            if self.maybe_unused_trait_imports.contains(&id) {
                // Check later.
                return;
            }
            self.unused_import(self.base_id).add(id);
        } else {
            // This trait import is definitely used, in a way other than
            // method resolution.
            self.maybe_unused_trait_imports.remove(&id);
            if let Some(i) = self.unused_imports.get_mut(&self.base_id) {
                i.unused.remove(&id);
            }
        }
    }

    fn unused_import(&mut self, id: ast::NodeId) -> &mut UnusedImport<'a> {
        let use_tree_id = self.base_id;
        let use_tree = self.base_use_tree.unwrap();
        let item_span = self.item_span;

        self.unused_imports
            .entry(id)
            .or_insert_with(|| UnusedImport {
                use_tree,
                use_tree_id,
                item_span,
                unused: FxHashSet::default(),
            })
    }
}

impl<'a, 'b> Visitor<'a> for UnusedImportCheckVisitor<'a, 'b> {
    fn visit_item(&mut self, item: &'a ast::Item) {
        self.item_span = item.span;

        // Ignore is_public import statements because there's no way to be sure
        // whether they're used or not. Also ignore imports with a dummy span
        // because this means that they were generated in some fashion by the
        // compiler and we don't need to consider them.
        if let ast::ItemKind::Use(..) = item.node {
            if item.vis.node.is_pub() || item.span.is_dummy() {
                return;
            }
        }

        visit::walk_item(self, item);
    }

    fn visit_use_tree(&mut self, use_tree: &'a ast::UseTree, id: ast::NodeId, nested: bool) {
        // Use the base UseTree's NodeId as the item id
        // This allows the grouping of all the lints in the same item
        if !nested {
            self.base_id = id;
            self.base_use_tree = Some(use_tree);
        }

        if let ast::UseTreeKind::Nested(ref items) = use_tree.kind {
            if items.is_empty() {
                self.unused_import(self.base_id).add(id);
            }
        } else {
            self.check_import(id);
        }

        visit::walk_use_tree(self, use_tree, id);
    }
}

enum UnusedSpanResult {
    Used,
    FlatUnused(Span, Span),
    NestedFullUnused(Vec<Span>, Span),
    NestedPartialUnused(Vec<Span>, Vec<Span>),
}

fn calc_unused_spans(
    unused_import: &UnusedImport<'_>,
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
                UnusedSpanResult::FlatUnused(use_tree.span, full_span)
            } else {
                UnusedSpanResult::Used
            }
        }
        ast::UseTreeKind::Nested(ref nested) => {
            if nested.len() == 0 {
                return UnusedSpanResult::FlatUnused(use_tree.span, full_span);
            }

            let mut unused_spans = Vec::new();
            let mut to_remove = Vec::new();
            let mut all_nested_unused = true;
            let mut previous_unused = false;
            for (pos, (use_tree, use_tree_id)) in nested.iter().enumerate() {
                let remove = match calc_unused_spans(unused_import, use_tree, *use_tree_id) {
                    UnusedSpanResult::Used => {
                        all_nested_unused = false;
                        None
                    }
                    UnusedSpanResult::FlatUnused(span, remove) => {
                        unused_spans.push(span);
                        Some(remove)
                    }
                    UnusedSpanResult::NestedFullUnused(mut spans, remove) => {
                        unused_spans.append(&mut spans);
                        Some(remove)
                    }
                    UnusedSpanResult::NestedPartialUnused(mut spans, mut to_remove_extra) => {
                        all_nested_unused = false;
                        unused_spans.append(&mut spans);
                        to_remove.append(&mut to_remove_extra);
                        None
                    }
                };
                if let Some(remove) = remove {
                    let remove_span = if nested.len() == 1 {
                        remove
                    } else if pos == nested.len() - 1 || !all_nested_unused {
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
                previous_unused = remove.is_some();
            }
            if unused_spans.is_empty() {
                UnusedSpanResult::Used
            } else if all_nested_unused {
                UnusedSpanResult::NestedFullUnused(unused_spans, full_span)
            } else {
                UnusedSpanResult::NestedPartialUnused(unused_spans, to_remove)
            }
        }
    }
}

pub fn check_crate(resolver: &mut Resolver<'_>, krate: &ast::Crate) {
    for directive in resolver.potentially_unused_imports.iter() {
        match directive.subclass {
            _ if directive.used.get() ||
                 directive.vis.get() == ty::Visibility::Public ||
                 directive.span.is_dummy() => {
                if let ImportDirectiveSubclass::MacroUse = directive.subclass {
                    if !directive.span.is_dummy() {
                        resolver.session.buffer_lint(
                            lint::builtin::MACRO_USE_EXTERN_CRATE,
                            directive.id,
                            directive.span,
                            "deprecated `#[macro_use]` directive used to \
                             import macros should be replaced at use sites \
                             with a `use` statement to import the macro \
                             instead",
                        );
                    }
                }
            }
            ImportDirectiveSubclass::ExternCrate { .. } => {
                resolver.maybe_unused_extern_crates.push((directive.id, directive.span));
            }
            ImportDirectiveSubclass::MacroUse => {
                let lint = lint::builtin::UNUSED_IMPORTS;
                let msg = "unused `#[macro_use]` import";
                resolver.session.buffer_lint(lint, directive.id, directive.span, msg);
            }
            _ => {}
        }
    }

    for (id, span) in resolver.unused_labels.iter() {
        resolver.session.buffer_lint(lint::builtin::UNUSED_LABELS, *id, *span, "unused label");
    }

    let mut visitor = UnusedImportCheckVisitor {
        resolver,
        unused_imports: Default::default(),
        base_use_tree: None,
        base_id: ast::DUMMY_NODE_ID,
        item_span: DUMMY_SP,
    };
    visit::walk_crate(&mut visitor, krate);

    for unused in visitor.unused_imports.values() {
        let mut fixes = Vec::new();
        let mut spans = match calc_unused_spans(unused, unused.use_tree, unused.use_tree_id) {
            UnusedSpanResult::Used => continue,
            UnusedSpanResult::FlatUnused(span, remove) => {
                fixes.push((remove, String::new()));
                vec![span]
            }
            UnusedSpanResult::NestedFullUnused(spans, remove) => {
                fixes.push((remove, String::new()));
                spans
            }
            UnusedSpanResult::NestedPartialUnused(spans, remove) => {
                for fix in &remove {
                    fixes.push((*fix, String::new()));
                }
                spans
            }
        };

        let len = spans.len();
        spans.sort();
        let ms = MultiSpan::from_spans(spans.clone());
        let mut span_snippets = spans.iter()
            .filter_map(|s| {
                match visitor.session.source_map().span_to_snippet(*s) {
                    Ok(s) => Some(format!("`{}`", s)),
                    _ => None,
                }
            }).collect::<Vec<String>>();
        span_snippets.sort();
        let msg = format!("unused import{}{}",
                          if len > 1 { "s" } else { "" },
                          if !span_snippets.is_empty() {
                              format!(": {}", span_snippets.join(", "))
                          } else {
                              String::new()
                          });

        let fix_msg = if fixes.len() == 1 && fixes[0].0 == unused.item_span {
            "remove the whole `use` item"
        } else if spans.len() > 1 {
            "remove the unused imports"
        } else {
            "remove the unused import"
        };

        visitor.session.buffer_lint_with_diagnostic(
            lint::builtin::UNUSED_IMPORTS,
            unused.use_tree_id,
            ms,
            &msg,
            lint::builtin::BuiltinLintDiagnostics::UnusedImports(fix_msg.into(), fixes),
        );
    }
}
