// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


//
// Unused import checking
//
// Although this is mostly a lint pass, it lives in here because it depends on
// resolve data structures and because it finalises the privacy information for
// `use` directives.
//
// Unused trait imports can't be checked until the method resolution. We save
// candidates here, and do the acutal check in librustc_typeck/check_unused.rs.

use std::ops::{Deref, DerefMut};

use Resolver;
use resolve_imports::ImportDirectiveSubclass;

use rustc::{lint, ty};
use rustc::util::nodemap::NodeMap;
use syntax::ast::{self, ViewPathGlob, ViewPathList, ViewPathSimple};
use syntax::visit::{self, Visitor};
use syntax_pos::{Span, MultiSpan, DUMMY_SP};


struct UnusedImportCheckVisitor<'a, 'b: 'a> {
    resolver: &'a mut Resolver<'b>,
    /// All the (so far) unused imports, grouped path list
    unused_imports: NodeMap<NodeMap<Span>>,
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
            self.unused_imports.entry(item_id).or_insert_with(NodeMap).insert(id, span);
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
        visit::walk_item(self, item);
        // Ignore is_public import statements because there's no way to be sure
        // whether they're used or not. Also ignore imports with a dummy span
        // because this means that they were generated in some fashion by the
        // compiler and we don't need to consider them.
        if item.vis == ast::Visibility::Public || item.span.source_equal(&DUMMY_SP) {
            return;
        }

        match item.node {
            ast::ItemKind::Use(ref p) => {
                match p.node {
                    ViewPathSimple(..) => {
                        self.check_import(item.id, item.id, p.span)
                    }

                    ViewPathList(_, ref list) => {
                        if list.len() == 0 {
                            self.unused_imports
                                .entry(item.id)
                                .or_insert_with(NodeMap)
                                .insert(item.id, item.span);
                        }
                        for i in list {
                            self.check_import(item.id, i.node.id, i.span);
                        }
                    }
                    ViewPathGlob(_) => {
                        self.check_import(item.id, item.id, p.span);
                    }
                }
            }
            _ => {}
        }
    }
}

pub fn check_crate(resolver: &mut Resolver, krate: &ast::Crate) {
    for directive in resolver.potentially_unused_imports.iter() {
        match directive.subclass {
            _ if directive.used.get() ||
                 directive.vis.get() == ty::Visibility::Public ||
                 directive.span.source_equal(&DUMMY_SP) => {}
            ImportDirectiveSubclass::ExternCrate => {
                let lint = lint::builtin::UNUSED_EXTERN_CRATES;
                let msg = "unused extern crate".to_string();
                resolver.session.add_lint(lint, directive.id, directive.span, msg);
            }
            ImportDirectiveSubclass::MacroUse => {
                let lint = lint::builtin::UNUSED_IMPORTS;
                let msg = "unused `#[macro_use]` import".to_string();
                resolver.session.add_lint(lint, directive.id, directive.span, msg);
            }
            _ => {}
        }
    }

    let mut visitor = UnusedImportCheckVisitor {
        resolver: resolver,
        unused_imports: NodeMap(),
    };
    visit::walk_crate(&mut visitor, krate);

    for (id, spans) in &visitor.unused_imports {
        let len = spans.len();
        let mut spans = spans.values().map(|s| *s).collect::<Vec<Span>>();
        spans.sort();
        let ms = MultiSpan::from_spans(spans.clone());
        let mut span_snippets = spans.iter()
            .filter_map(|s| {
                match visitor.session.codemap().span_to_snippet(*s) {
                    Ok(s) => Some(format!("`{}`", s)),
                    _ => None,
                }
            }).collect::<Vec<String>>();
        span_snippets.sort();
        let msg = format!("unused import{}{}",
                          if len > 1 { "s" } else { "" },
                          if span_snippets.len() > 0 {
                              format!(": {}", span_snippets.join(", "))
                          } else {
                              String::new()
                          });
        visitor.session.add_lint(lint::builtin::UNUSED_IMPORTS,
                                 *id,
                                 ms,
                                 msg);
    }
}
