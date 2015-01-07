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

use std::ops::{Deref, DerefMut};

use Resolver;
use Namespace::{TypeNS, ValueNS};

use rustc::lint;
use rustc::middle::privacy::{DependsOn, LastImport, Used, Unused};
use syntax::ast;
use syntax::ast::{ViewItem, ViewItemExternCrate, ViewItemUse};
use syntax::ast::{ViewPathGlob, ViewPathList, ViewPathSimple};
use syntax::codemap::{Span, DUMMY_SP};
use syntax::visit::{self, Visitor};

struct UnusedImportCheckVisitor<'a, 'b:'a, 'tcx:'b> {
    resolver: &'a mut Resolver<'b, 'tcx>
}

// Deref and DerefMut impls allow treating UnusedImportCheckVisitor as Resolver.
impl<'a, 'b, 'tcx:'b> Deref for UnusedImportCheckVisitor<'a, 'b, 'tcx> {
    type Target = Resolver<'b, 'tcx>;

    fn deref<'c>(&'c self) -> &'c Resolver<'b, 'tcx> {
        &*self.resolver
    }
}

impl<'a, 'b, 'tcx:'b> DerefMut for UnusedImportCheckVisitor<'a, 'b, 'tcx> {
    fn deref_mut<'c>(&'c mut self) -> &'c mut Resolver<'b, 'tcx> {
        &mut *self.resolver
    }
}

impl<'a, 'b, 'tcx> UnusedImportCheckVisitor<'a, 'b, 'tcx> {
    // We have information about whether `use` (import) directives are actually used now.
    // If an import is not used at all, we signal a lint error. If an import is only used
    // for a single namespace, we remove the other namespace from the recorded privacy
    // information. That means in privacy.rs, we will only check imports and namespaces
    // which are used. In particular, this means that if an import could name either a
    // public or private item, we will check the correct thing, dependent on how the import
    // is used.
    fn finalize_import(&mut self, id: ast::NodeId, span: Span) {
        debug!("finalizing import uses for {:?}",
                self.session.codemap().span_to_snippet(span));

        if !self.used_imports.contains(&(id, TypeNS)) &&
           !self.used_imports.contains(&(id, ValueNS)) {
            self.session.add_lint(lint::builtin::UNUSED_IMPORTS,
                                  id,
                                  span,
                                  "unused import".to_string());
        }

        let (v_priv, t_priv) = match self.last_private.get(&id) {
            Some(&LastImport {
                value_priv: v,
                value_used: _,
                type_priv: t,
                type_used: _
            }) => (v, t),
            Some(_) => {
                panic!("we should only have LastImport for `use` directives")
            }
            _ => return,
        };

        let mut v_used = if self.used_imports.contains(&(id, ValueNS)) {
            Used
        } else {
            Unused
        };
        let t_used = if self.used_imports.contains(&(id, TypeNS)) {
            Used
        } else {
            Unused
        };

        match (v_priv, t_priv) {
            // Since some items may be both in the value _and_ type namespaces (e.g., structs)
            // we might have two LastPrivates pointing at the same thing. There is no point
            // checking both, so lets not check the value one.
            (Some(DependsOn(def_v)), Some(DependsOn(def_t))) if def_v == def_t => v_used = Unused,
            _ => {},
        }

        self.last_private.insert(id, LastImport{value_priv: v_priv,
                                                value_used: v_used,
                                                type_priv: t_priv,
                                                type_used: t_used});
    }
}

impl<'a, 'b, 'v, 'tcx> Visitor<'v> for UnusedImportCheckVisitor<'a, 'b, 'tcx> {
    fn visit_view_item(&mut self, vi: &ViewItem) {
        // Ignore is_public import statements because there's no way to be sure
        // whether they're used or not. Also ignore imports with a dummy span
        // because this means that they were generated in some fashion by the
        // compiler and we don't need to consider them.
        if vi.vis == ast::Public || vi.span == DUMMY_SP {
            visit::walk_view_item(self, vi);
            return;
        }

        match vi.node {
            ViewItemExternCrate(_, _, id) => {
                if let Some(crate_num) = self.session.cstore.find_extern_mod_stmt_cnum(id) {
                    if !self.used_crates.contains(&crate_num) {
                        self.session.add_lint(lint::builtin::UNUSED_EXTERN_CRATES,
                                              id,
                                              vi.span,
                                              "unused extern crate".to_string());
                    }
                }
            },
            ViewItemUse(ref p) => {
                match p.node {
                    ViewPathSimple(_, _, id) => {
                        self.finalize_import(id, p.span)
                    }

                    ViewPathList(_, ref list, _) => {
                        for i in list.iter() {
                            self.finalize_import(i.node.id(), i.span);
                        }
                    }
                    ViewPathGlob(_, id) => {
                        if !self.used_imports.contains(&(id, TypeNS)) &&
                           !self.used_imports.contains(&(id, ValueNS)) {
                            self.session
                                .add_lint(lint::builtin::UNUSED_IMPORTS,
                                          id,
                                          p.span,
                                          "unused import".to_string());
                        }
                    }
                }
            }
        }

        visit::walk_view_item(self, vi);
    }
}

pub fn check_crate(resolver: &mut Resolver, krate: &ast::Crate) {
    let mut visitor = UnusedImportCheckVisitor { resolver: resolver };
    visit::walk_crate(&mut visitor, krate);
}
