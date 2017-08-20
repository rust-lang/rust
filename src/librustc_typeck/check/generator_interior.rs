// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use log;
use rustc::hir::def_id::DefId;
use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};
use rustc::hir::{self, Body, Pat, PatKind, Expr};
use rustc::middle::region::{RegionMaps, CodeExtent};
use rustc::ty::Ty;
use syntax::ast::NodeId;
use syntax::codemap::Span;
use std::rc::Rc;
use super::FnCtxt;
use util::nodemap::FxHashMap;

struct InteriorVisitor<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    fcx: &'a FnCtxt<'a, 'gcx, 'tcx>,
    cache: FxHashMap<NodeId, Option<Span>>,
    types: FxHashMap<Ty<'tcx>, usize>,
    region_maps: Rc<RegionMaps>,
}

impl<'a, 'gcx, 'tcx> InteriorVisitor<'a, 'gcx, 'tcx> {
    fn record(&mut self, ty: Ty<'tcx>, scope: Option<CodeExtent>, expr: Option<&'tcx Expr>) {
        use syntax_pos::DUMMY_SP;

        let live_across_yield = scope.map(|s| {
            self.fcx.tcx.yield_in_extent(s, &mut self.cache).is_some()
        }).unwrap_or(true);

        if live_across_yield {
            let ty = self.fcx.resolve_type_vars_if_possible(&ty);

            if log_enabled!(log::LogLevel::Debug) {
                let span = scope.map(|s| s.span(&self.fcx.tcx.hir).unwrap_or(DUMMY_SP));
                debug!("type in expr = {:?}, scope = {:?}, type = {:?}, span = {:?}",
                       expr, scope, ty, span);
            }

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
        cache: FxHashMap(),
        region_maps: fcx.tcx.region_maps(def_id),
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
