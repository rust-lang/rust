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
use rustc::ty::TyCtxt;

use syntax::ast;
use syntax_pos::{Span, DUMMY_SP};

use rustc::hir::def_id::LOCAL_CRATE;
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::hir;
use rustc::util::nodemap::DefIdSet;

struct CheckVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    used_trait_imports: DefIdSet,
}

impl<'a, 'tcx> CheckVisitor<'a, 'tcx> {
    fn check_import(&self, id: ast::NodeId, span: Span) {
        let def_id = self.tcx.hir.local_def_id(id);
        if !self.tcx.maybe_unused_trait_import(def_id) {
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
        self.tcx.lint_node(lint::builtin::UNUSED_IMPORTS, id, span, &msg);
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
    let mut used_trait_imports = DefIdSet();
    for &body_id in tcx.hir.krate().bodies.keys() {
        let item_def_id = tcx.hir.body_owner_def_id(body_id);
        let imports = tcx.used_trait_imports(item_def_id);
        debug!("GatherVisitor: item_def_id={:?} with imports {:#?}", item_def_id, imports);
        used_trait_imports.extend(imports.iter());
    }

    let mut visitor = CheckVisitor { tcx, used_trait_imports };
    tcx.hir.krate().visit_all_item_likes(&mut visitor);

    for &(def_id, span) in tcx.maybe_unused_extern_crates(LOCAL_CRATE).iter() {
        // The `def_id` here actually was calculated during resolution (at least
        // at the time of this writing) and is being shipped to us via a side
        // channel of the tcx. There may have been extra expansion phases,
        // however, which ended up removing the `def_id` *after* expansion such
        // as the `ReplaceBodyWithLoop` pass (which is a bit of a hack, but hey)
        //
        // As a result we need to verify that `def_id` is indeed still valid for
        // our AST and actually present in the HIR map. If it's not there then
        // there's safely nothing to warn about, and otherwise we carry on with
        // our execution.
        //
        // Note that if we carry through to the `extern_mod_stmt_cnum` query
        // below it'll cause a panic because `def_id` is actually bogus at this
        // point in time otherwise.
        if let Some(id) = tcx.hir.as_local_node_id(def_id) {
            if tcx.hir.find(id).is_none() {
                continue
            }
        }
        let cnum = tcx.extern_mod_stmt_cnum(def_id).unwrap();
        if tcx.is_compiler_builtins(cnum) {
            continue
        }
        if tcx.is_panic_runtime(cnum) {
            continue
        }
        if tcx.has_global_allocator(cnum) {
            continue
        }
        assert_eq!(def_id.krate, LOCAL_CRATE);
        let hir_id = tcx.hir.definitions().def_index_to_hir_id(def_id.index);
        let id = tcx.hir.definitions().find_node_for_hir_id(hir_id);
        let lint = lint::builtin::UNUSED_EXTERN_CRATES;
        let msg = "unused extern crate";
        tcx.lint_node(lint, id, span, msg);
    }
}
