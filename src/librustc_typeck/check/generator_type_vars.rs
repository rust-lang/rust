// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};
use rustc::hir::{self, Pat, Expr};
use super::FnCtxt;
use syntax::ast::NodeId;
use syntax_pos::Span;
use rustc::ty::fold::TypeFoldable;

struct InferVarVisitor<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    fcx: &'a FnCtxt<'a, 'gcx, 'tcx>,
}

impl<'a, 'gcx, 'tcx> InferVarVisitor<'a, 'gcx, 'tcx> {
    fn visit_node_id(&mut self, span: Span, hir_id: hir::HirId) {
        // Resolve any borrowings for the node with id `node_id`
        self.visit_adjustments(span, hir_id);

        // Resolve the type of the node with id `node_id`
        let n_ty = self.fcx.node_ty(hir_id);
        self.resolve(&n_ty, span);

        // Resolve any substitutions
        if let Some(substs) = self.fcx.tables.borrow().node_substs_opt(hir_id) {
            self.resolve(&substs, span)
        }

        if let Some(ftys) = self.fcx.tables.borrow().fru_field_types().get(hir_id) {
            self.resolve(ftys, span)
        }
    }

    fn resolve<T: TypeFoldable<'tcx>>(&self, x: &T, span: Span) {
        if self.fcx.any_unresolved_type_vars(x) {
            self.fcx.tcx.sess.span_warn(span, "infer variable in generator, please report this");
        }
    }

    fn visit_adjustments(&mut self, span: Span, hir_id: hir::HirId) {
        let adjustments = self.fcx.tables.borrow_mut();
        let adjustments = adjustments.adjustments();
        if let Some(adjustments) = adjustments.get(hir_id) {
            self.resolve(adjustments, span);
        }
    }
}

pub fn find_type_vars<'a, 'gcx, 'tcx>(fcx: &'a FnCtxt<'a, 'gcx, 'tcx>,
                                      node_id: NodeId,
                                      body_id: hir::BodyId) {
    let body = fcx.tcx.hir.body(body_id);
    let mut visitor = InferVarVisitor {
        fcx,
    };
    for arg in &body.arguments {
        visitor.visit_node_id(arg.pat.span, arg.hir_id);
    }

    let hir_id = fcx.tcx.hir.node_to_hir_id(node_id);
    let gen_sig = fcx.tables.borrow().generator_sigs().get(hir_id).unwrap().unwrap();
    visitor.resolve(&gen_sig.yield_ty, body.value.span);
    visitor.resolve(&gen_sig.return_ty, body.value.span);

    visitor.visit_body(body);
}

impl<'a, 'gcx, 'tcx> Visitor<'tcx> for InferVarVisitor<'a, 'gcx, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_block(&mut self, b: &'tcx hir::Block) {
        self.visit_node_id(b.span, b.hir_id);
        intravisit::walk_block(self, b);
    }

    fn visit_local(&mut self, l: &'tcx hir::Local) {
        intravisit::walk_local(self, l);
        let var_ty = self.fcx.local_ty(l.span, l.id);
        self.resolve(&var_ty, l.span);
    }

    fn visit_ty(&mut self, hir_ty: &'tcx hir::Ty) {
        intravisit::walk_ty(self, hir_ty);
        let ty = self.fcx.node_ty(hir_ty.hir_id);
        self.resolve(&ty, hir_ty.span);
    }

    fn visit_pat(&mut self, pat: &'tcx Pat) {
        self.visit_node_id(pat.span, pat.hir_id);
        intravisit::walk_pat(self, pat);
    }

    fn visit_expr(&mut self, expr: &'tcx Expr) {
        self.visit_node_id(expr.span, expr.hir_id);
        intravisit::walk_expr(self, expr);
    }
}
