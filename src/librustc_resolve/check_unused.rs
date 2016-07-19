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
use Namespace::{TypeNS, ValueNS};

use rustc::lint;
use syntax::ast::{self, ViewPathGlob, ViewPathList, ViewPathSimple};
use syntax::visit::{self, Visitor};
use syntax_pos::{Span, DUMMY_SP};


struct UnusedImportCheckVisitor<'a, 'b: 'a> {
    resolver: &'a mut Resolver<'b>,
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
    fn check_import(&mut self, id: ast::NodeId, span: Span) {
        if !self.used_imports.contains(&(id, TypeNS)) &&
           !self.used_imports.contains(&(id, ValueNS)) {
            if self.maybe_unused_trait_imports.contains(&id) {
                // Check later.
                return;
            }
            self.session.add_lint(lint::builtin::UNUSED_IMPORTS,
                                  id,
                                  span,
                                  "unused import".to_string());
        } else {
            // This trait import is definitely used, in a way other than
            // method resolution.
            self.maybe_unused_trait_imports.remove(&id);
        }
    }
}

impl<'a, 'b> Visitor for UnusedImportCheckVisitor<'a, 'b> {
    fn visit_item(&mut self, item: &ast::Item) {
        visit::walk_item(self, item);
        // Ignore is_public import statements because there's no way to be sure
        // whether they're used or not. Also ignore imports with a dummy span
        // because this means that they were generated in some fashion by the
        // compiler and we don't need to consider them.
        if item.vis == ast::Visibility::Public || item.span.source_equal(&DUMMY_SP) {
            return;
        }

        match item.node {
            ast::ItemKind::ExternCrate(_) => {
                if let Some(crate_num) = self.session.cstore.extern_mod_stmt_cnum(item.id) {
                    if !self.used_crates.contains(&crate_num) {
                        self.session.add_lint(lint::builtin::UNUSED_EXTERN_CRATES,
                                              item.id,
                                              item.span,
                                              "unused extern crate".to_string());
                    }
                }
            }
            ast::ItemKind::Use(ref p) => {
                match p.node {
                    ViewPathSimple(_, _) => {
                        self.check_import(item.id, p.span)
                    }

                    ViewPathList(_, ref list) => {
                        for i in list {
                            self.check_import(i.node.id(), i.span);
                        }
                    }
                    ViewPathGlob(_) => {
                        self.check_import(item.id, p.span)
                    }
                }
            }
            _ => {}
        }
    }
}

pub fn check_crate(resolver: &mut Resolver, krate: &ast::Crate) {
    let mut visitor = UnusedImportCheckVisitor { resolver: resolver };
    visit::walk_crate(&mut visitor, krate);
}
