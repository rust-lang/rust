// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lint;
use rustc::dep_graph::DepNode;
use rustc::ty::TyCtxt;

use syntax::ast;
use syntax_pos::{Span, DUMMY_SP};

use rustc::hir;
use rustc::hir::itemlikevisit::ItemLikeVisitor;

struct UnusedTraitImportVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
}

impl<'a, 'tcx> UnusedTraitImportVisitor<'a, 'tcx> {
    fn check_import(&self, id: ast::NodeId, span: Span) {
        if !self.tcx.maybe_unused_trait_imports.contains(&id) {
            return;
        }
        if self.tcx.used_trait_imports.borrow().contains(&id) {
            return;
        }

        let msg = if let Ok(snippet) = self.tcx.sess.codemap().span_to_snippet(span) {
            format!("unused import: `{}`", snippet)
        } else {
            "unused import".to_string()
        };
        self.tcx.sess.add_lint(lint::builtin::UNUSED_IMPORTS, id, span, msg);
    }
}

impl<'a, 'tcx, 'v> ItemLikeVisitor<'v> for UnusedTraitImportVisitor<'a, 'tcx> {
    fn visit_item(&mut self, item: &hir::Item) {
        if item.vis == hir::Public || item.span == DUMMY_SP {
            return;
        }
        if let hir::ItemUse(ref path, _) = item.node {
            self.check_import(item.id, path.span);
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem) {
    }

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem) {
    }
}

pub fn check_crate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    let _task = tcx.dep_graph.in_task(DepNode::UnusedTraitCheck);
    let mut visitor = UnusedTraitImportVisitor { tcx: tcx };
    tcx.map.krate().visit_all_item_likes(&mut visitor);
}
