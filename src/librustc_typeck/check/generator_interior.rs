// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This calculates the types which has storage which lives across a suspension point in a
//! generator from the perspective of typeck. The actual types used at runtime
//! is calculated in `rustc_mir::transform::generator` and may be a subset of the
//! types computed here.

use rustc::hir::def_id::DefId;
use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};
use rustc::hir::{self, Pat, PatKind, Expr};
use rustc::middle::region;
use rustc::ty::Ty;
use std::rc::Rc;
use super::FnCtxt;
use util::nodemap::FxHashMap;

struct InteriorVisitor<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    fcx: &'a FnCtxt<'a, 'gcx, 'tcx>,
    types: FxHashMap<Ty<'tcx>, usize>,
    region_scope_tree: Rc<region::ScopeTree>,
    expr_count: usize,
}

impl<'a, 'gcx, 'tcx> InteriorVisitor<'a, 'gcx, 'tcx> {
    fn record(&mut self, ty: Ty<'tcx>, scope: Option<region::Scope>, expr: Option<&'tcx Expr>) {
        use syntax_pos::DUMMY_SP;

        if self.fcx.tcx.sess.verbose() {
        let span = scope.map_or(DUMMY_SP, |s| s.span(self.fcx.tcx, &self.region_scope_tree));
        self.fcx.tcx.sess.span_warn(span, &format!("temporary scope for node id {:?}", expr));
        }

        let live_across_yield = scope.map_or(Some(DUMMY_SP), |s| {
            self.region_scope_tree.yield_in_scope(s).and_then(|(span, expr_count)| {
                // Check if the span in the region comes after the expression
                if expr_count > self.expr_count {
                    Some(span)
                } else {
                    None
                }
            })
        });

        if let Some(span) = live_across_yield {
            let ty = self.fcx.resolve_type_vars_if_possible(&ty);

            debug!("type in expr = {:?}, scope = {:?}, type = {:?}, span = {:?}",
                   expr, scope, ty, span);

            // Map the type to the number of types added before it
            let entries = self.types.len();
            self.types.entry(&ty).or_insert(entries);
        } else {
            debug!("no type in expr = {:?}, span = {:?}", expr, expr.map(|e| e.span));
        }
    }
}

pub fn resolve_interior<'a, 'gcx, 'tcx>(fcx: &'a FnCtxt<'a, 'gcx, 'tcx>,
                                        def_id: DefId,
                                        body_id: hir::BodyId,
                                        witness: Ty<'tcx>) {
    let body = fcx.tcx.hir.body(body_id);
    let mut visitor = InteriorVisitor {
        fcx,
        types: FxHashMap(),
        region_scope_tree: fcx.tcx.region_scope_tree(def_id),
        expr_count: 0,
    };
    intravisit::walk_body(&mut visitor, body);

    let mut types: Vec<_> = visitor.types.drain().collect();

    // Sort types by insertion order
    types.sort_by_key(|t| t.1);

    // Extract type components
    let types: Vec<_> = types.into_iter().map(|t| t.0).collect();

    let tuple = fcx.tcx.intern_tup(&types, false);

    debug!("Types in generator {:?}, span = {:?}", tuple, body.value.span);

    // Unify the tuple with the witness
    match fcx.at(&fcx.misc(body.value.span), fcx.param_env).eq(witness, tuple) {
        Ok(ok) => fcx.register_infer_ok_obligations(ok),
        _ => bug!(),
   }
}

// This visitor has to have the same visit_expr calls as RegionResolutionVisitor in
// librustc/middle/region.rs since `expr_count` is compared against the results
// there.
impl<'a, 'gcx, 'tcx> Visitor<'tcx> for InteriorVisitor<'a, 'gcx, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_pat(&mut self, pat: &'tcx Pat) {
        if let PatKind::Binding(..) = pat.node {
            let scope = self.region_scope_tree.var_scope(pat.hir_id.local_id);
            let ty = self.fcx.tables.borrow().pat_ty(pat);
            self.record(ty, Some(scope), None);
        }

        intravisit::walk_pat(self, pat);
    }

    fn visit_expr(&mut self, expr: &'tcx Expr) {
        self.expr_count += 1;

        if self.fcx.tcx.sess.verbose() {
        self.fcx.tcx.sess.span_warn(expr.span, &format!("node id {:?}", expr.id));
        }

        let scope = self.region_scope_tree.temporary_scope(expr.hir_id.local_id);


        let ty = self.fcx.tables.borrow().expr_ty_adjusted(expr);
        self.record(ty, scope, Some(expr));

        intravisit::walk_expr(self, expr);
    }
}
