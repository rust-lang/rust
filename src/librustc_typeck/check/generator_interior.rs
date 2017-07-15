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
use rustc::hir::{self, Body, Pat, PatKind, Expr};
use rustc::hir::def_id::DefId;
use rustc::ty::Ty;
use rustc::middle::region::{RegionMaps, CodeExtent};
use util::nodemap::FxHashSet;
use std::rc::Rc;
use super::FnCtxt;

struct InteriorVisitor<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    fcx: &'a FnCtxt<'a, 'gcx, 'tcx>,
    types: FxHashSet<Ty<'tcx>>,
    region_maps: Rc<RegionMaps>,
}

impl<'a, 'gcx, 'tcx> InteriorVisitor<'a, 'gcx, 'tcx> {
    fn record(&mut self, ty: Ty<'tcx>, scope: Option<CodeExtent>, expr: Option<&'tcx Expr>) {
        use syntax_pos::DUMMY_SP;

        if scope.map(|s| self.fcx.tcx.yield_in_extent(s).is_some()).unwrap_or(true) {
            if self.fcx.tcx.sess.verbose() {
                if let Some(s) = scope {
                    self.fcx.tcx.sess.span_warn(s.span(&self.fcx.tcx.hir).unwrap_or(DUMMY_SP),
                        &format!("type in generator with scope = {:?}, type = {:?}",
                                 scope,
                                 self.fcx.resolve_type_vars_if_possible(&ty)));
                } else {
                    self.fcx.tcx.sess.span_warn(DUMMY_SP,
                        &format!("type in generator WITHOUT scope, type = {:?}",
                                 self.fcx.resolve_type_vars_if_possible(&ty)));
                }
                if let Some(e) = expr {
                    self.fcx.tcx.sess.span_warn(e.span,
                        &format!("type from expression: {:?}", e));
                }
            }
            self.types.insert(ty);
        } else if self.fcx.tcx.sess.verbose() {
            if let Some(e) = expr {
                self.fcx.tcx.sess.span_warn(e.span,
                    &format!("NO type from expression: {:?}", e));
            }
        }
    }
}

pub fn find_interior<'a, 'gcx, 'tcx>(fcx: &'a FnCtxt<'a, 'gcx, 'tcx>,
                                     def_id: DefId,
                                     body_id: hir::BodyId,
                                     witness: Ty<'tcx>) {
    let body = fcx.tcx.hir.body(body_id);
    let mut visitor = InteriorVisitor {
        fcx,
        types: FxHashSet(),
        region_maps: fcx.tcx.region_maps(def_id),
    };
    intravisit::walk_body(&mut visitor, body);

    // Deduplicate types
    let set: FxHashSet<_> = visitor.types.into_iter()
        .map(|t| fcx.resolve_type_vars_if_possible(&t))
        .collect();
    let types: Vec<_> = set.into_iter().collect();

    let tuple = fcx.tcx.intern_tup(&types, false);

    if fcx.tcx.sess.verbose() {
        fcx.tcx.sess.span_warn(body.value.span,
            &format!("Types in generator {:?}", tuple));
    }

    // Unify the tuple with the witness
    match fcx.at(&fcx.misc(body.value.span), fcx.param_env).eq(witness, tuple) {
        Ok(ok) => fcx.register_infer_ok_obligations(ok),
        _ => bug!(),
   }
}

impl<'a, 'gcx, 'tcx> Visitor<'tcx> for InteriorVisitor<'a, 'gcx, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_body(&mut self, _body: &'tcx Body) {
        // Closures inside are not considered part of the generator interior
    }

    fn visit_pat(&mut self, pat: &'tcx Pat) {
        if let PatKind::Binding(..) = pat.node {
            let scope = self.region_maps.var_scope(pat.id);
            let ty = self.fcx.tables.borrow().pat_ty(pat);
            self.record(ty, Some(scope), None);
        }

        intravisit::walk_pat(self, pat);
    }

    fn visit_expr(&mut self, expr: &'tcx Expr) {
        let scope = self.region_maps.temporary_scope(expr.id);
        let ty = self.fcx.tables.borrow().expr_ty_adjusted(expr);
        self.record(ty, scope, Some(expr));

        intravisit::walk_expr(self, expr);
    }
}
