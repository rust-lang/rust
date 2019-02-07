//
// Unused import checking
//
// Although this is mostly a lint pass, it lives in here because it depends on
// resolve data structures and because it finalises the privacy information for
// `use` directives.
//
// Unused trait imports can't be checked until the method resolution. We save
// candidates here, and do the actual check in librustc_typeck/check_unused.rs.

use std::ops::{Deref, DerefMut};

use crate::Resolver;
use crate::resolve_imports::ImportDirectiveSubclass;

use rustc::{lint, ty};
use rustc::util::nodemap::NodeMap;
use syntax::ast;
use syntax::visit::{self, Visitor};
use syntax_pos::{Span, MultiSpan, DUMMY_SP};


struct UnusedImportCheckVisitor<'a, 'b: 'a> {
    resolver: &'a mut Resolver<'b>,
    /// All the (so far) unused imports, grouped path list
    unused_imports: NodeMap<NodeMap<Span>>,
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
    fn check_import(&mut self, item_id: ast::NodeId, id: ast::NodeId, span: Span) {
        let mut used = false;
        self.per_ns(|this, ns| used |= this.used_imports.contains(&(id, ns)));
        if !used {
            if self.maybe_unused_trait_imports.contains(&id) {
                // Check later.
                return;
            }
            self.unused_imports.entry(item_id).or_default().insert(id, span);
        } else {
            // This trait import is definitely used, in a way other than
            // method resolution.
            self.maybe_unused_trait_imports.remove(&id);
            if let Some(i) = self.unused_imports.get_mut(&item_id) {
                i.remove(&id);
            }
        }
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
        }

        if let ast::UseTreeKind::Nested(ref items) = use_tree.kind {
            // If it's the parent group, cover the entire use item
            let span = if nested {
                use_tree.span
            } else {
                self.item_span
            };

            if items.is_empty() {
                self.unused_imports
                    .entry(self.base_id)
                    .or_default()
                    .insert(id, span);
            }
        } else {
            let base_id = self.base_id;
            self.check_import(base_id, id, use_tree.span);
        }

        visit::walk_use_tree(self, use_tree, id);
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
        base_id: ast::DUMMY_NODE_ID,
        item_span: DUMMY_SP,
    };
    visit::walk_crate(&mut visitor, krate);

    for (id, spans) in &visitor.unused_imports {
        let len = spans.len();
        let mut spans = spans.values().cloned().collect::<Vec<Span>>();
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
        visitor.session.buffer_lint(lint::builtin::UNUSED_IMPORTS, *id, ms, &msg);
    }
}
