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
use rustc::util::nodemap::DefIdSet;

struct CheckVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    used_trait_imports: DefIdSet,
}

impl<'a, 'tcx> CheckVisitor<'a, 'tcx> {
    fn check_import(&self, id: ast::NodeId, span: Span) {
        if !self.tcx.maybe_unused_trait_imports.contains(&id) {
            return;
        }

        let import_def_id = self.tcx.hir.local_def_id(id);
        if self.used_trait_imports.contains(&import_def_id) {
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

impl<'a, 'tcx, 'v> ItemLikeVisitor<'v> for CheckVisitor<'a, 'tcx> {
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

    let mut used_trait_imports = DefIdSet();
    for &body_id in tcx.hir.krate().bodies.keys() {
        let item_id = tcx.hir.body_owner(body_id);
        let item_def_id = tcx.hir.local_def_id(item_id);

        // this will have been written by the main typeck pass
        if let Some(imports) = tcx.used_trait_imports.borrow().get(&item_def_id) {
            debug!("GatherVisitor: item_def_id={:?} with imports {:#?}", item_def_id, imports);
            used_trait_imports.extend(imports);
        } else {
            debug!("GatherVisitor: item_def_id={:?} with no imports", item_def_id);
        }
    }

    let mut visitor = CheckVisitor { tcx, used_trait_imports };
    tcx.hir.krate().visit_all_item_likes(&mut visitor);
}
