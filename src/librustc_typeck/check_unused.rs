// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lint;
use middle::ty;

use syntax::ast;
use syntax::codemap::{Span, DUMMY_SP};

use rustc_front::hir;
use rustc_front::intravisit::Visitor;

struct UnusedTraitImportVisitor<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
}

impl<'a, 'tcx> UnusedTraitImportVisitor<'a, 'tcx> {
    fn check_import(&self, id: ast::NodeId, span: Span) {
        if !self.tcx.maybe_unused_trait_imports.contains(&id) {
            return;
        }
        if self.tcx.used_trait_imports.borrow().contains(&id) {
            return;
        }
        self.tcx.sess.add_lint(lint::builtin::UNUSED_IMPORTS,
                               id,
                               span,
                               "unused import".to_string());
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for UnusedTraitImportVisitor<'a, 'tcx> {
    fn visit_item(&mut self, item: &hir::Item) {
        if item.vis == hir::Public || item.span == DUMMY_SP {
            return;
        }
        if let hir::ItemUse(ref path) = item.node {
            match path.node {
                hir::ViewPathSimple(..) | hir::ViewPathGlob(..) => {
                    self.check_import(item.id, path.span);
                }
                hir::ViewPathList(_, ref path_list) => {
                    for path_item in path_list {
                        self.check_import(path_item.node.id(), path_item.span);
                    }
                }
            }
        }
    }
}

pub fn check_crate(tcx: &ty::ctxt) {
    let mut visitor = UnusedTraitImportVisitor { tcx: tcx };
    tcx.map.krate().visit_all_items(&mut visitor);
}
