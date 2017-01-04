// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Type resolution: the phase that finds all the types in the AST with
// unresolved type variables and replaces "ty_var" types with their
// substitutions.
use self::ResolveReason::*;

use check::FnCtxt;
use hir::def_id::DefId;
use rustc::ty::{self, Ty, TyCtxt, MethodCall, MethodCallee};
use rustc::ty::adjustment;
use rustc::ty::fold::{TypeFolder,TypeFoldable};
use rustc::infer::{InferCtxt, FixupError};
use rustc::util::nodemap::DefIdMap;

use std::cell::Cell;

use syntax::ast;
use syntax_pos::Span;

use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};
use rustc::hir;

///////////////////////////////////////////////////////////////////////////
// Entry point

impl<'a, 'gcx, 'tcx> FnCtxt<'a, 'gcx, 'tcx> {
    pub fn resolve_type_vars_in_body(&self, body: &'gcx hir::Body) {
        assert_eq!(self.writeback_errors.get(), false);

        let item_id = self.tcx.map.body_owner(body.id());
        let item_def_id = self.tcx.map.local_def_id(item_id);

        let mut wbcx = WritebackCx::new(self);
        for arg in &body.arguments {
            wbcx.visit_node_id(ResolvingPattern(arg.pat.span), arg.id);
        }
        wbcx.visit_body(body);
        wbcx.visit_upvar_borrow_map();
        wbcx.visit_closures();
        wbcx.visit_liberated_fn_sigs();
        wbcx.visit_fru_field_types();
        wbcx.visit_anon_types();
        wbcx.visit_deferred_obligations(item_id);
        wbcx.visit_type_nodes();

        let tables = self.tcx.alloc_tables(wbcx.tables);
        self.tcx.tables.borrow_mut().insert(item_def_id, tables);
    }
}

///////////////////////////////////////////////////////////////////////////
// The Writerback context. This visitor walks the AST, checking the
// fn-specific tables to find references to types or regions. It
// resolves those regions to remove inference variables and writes the
// final result back into the master tables in the tcx. Here and
// there, it applies a few ad-hoc checks that were not convenient to
// do elsewhere.

struct WritebackCx<'cx, 'gcx: 'cx+'tcx, 'tcx: 'cx> {
    fcx: &'cx FnCtxt<'cx, 'gcx, 'tcx>,

    tables: ty::Tables<'gcx>,

    // Mapping from free regions of the function to the
    // early-bound versions of them, visible from the
    // outside of the function. This is needed by, and
    // only populated if there are any `impl Trait`.
    free_to_bound_regions: DefIdMap<&'gcx ty::Region>
}

impl<'cx, 'gcx, 'tcx> WritebackCx<'cx, 'gcx, 'tcx> {
    fn new(fcx: &'cx FnCtxt<'cx, 'gcx, 'tcx>) -> WritebackCx<'cx, 'gcx, 'tcx> {
        let mut wbcx = WritebackCx {
            fcx: fcx,
            tables: ty::Tables::empty(),
            free_to_bound_regions: DefIdMap()
        };

        // Only build the reverse mapping if `impl Trait` is used.
        if fcx.anon_types.borrow().is_empty() {
            return wbcx;
        }

        let gcx = fcx.tcx.global_tcx();
        let free_substs = fcx.parameter_environment.free_substs;
        for (i, k) in free_substs.params().iter().enumerate() {
            let r = if let Some(r) = k.as_region() {
                r
            } else {
                continue;
            };
            match *r {
                ty::ReFree(ty::FreeRegion {
                    bound_region: ty::BoundRegion::BrNamed(def_id, name, _), ..
                }) => {
                    let bound_region = gcx.mk_region(ty::ReEarlyBound(ty::EarlyBoundRegion {
                        index: i as u32,
                        name: name,
                    }));
                    wbcx.free_to_bound_regions.insert(def_id, bound_region);
                }
                _ => {
                    bug!("{:?} is not a free region for an early-bound lifetime", r);
                }
            }
        }

        wbcx
    }

    fn tcx(&self) -> TyCtxt<'cx, 'gcx, 'tcx> {
        self.fcx.tcx
    }

    fn write_ty_to_tables(&mut self, node_id: ast::NodeId, ty: Ty<'gcx>) {
        debug!("write_ty_to_tables({}, {:?})", node_id,  ty);
        assert!(!ty.needs_infer());
        self.tables.node_types.insert(node_id, ty);
    }

    // Hacky hack: During type-checking, we treat *all* operators
    // as potentially overloaded. But then, during writeback, if
    // we observe that something like `a+b` is (known to be)
    // operating on scalars, we clear the overload.
    fn fix_scalar_builtin_expr(&mut self, e: &hir::Expr) {
        match e.node {
            hir::ExprUnary(hir::UnNeg, ref inner) |
            hir::ExprUnary(hir::UnNot, ref inner)  => {
                let inner_ty = self.fcx.node_ty(inner.id);
                let inner_ty = self.fcx.resolve_type_vars_if_possible(&inner_ty);

                if inner_ty.is_scalar() {
                    self.fcx.tables.borrow_mut().method_map.remove(&MethodCall::expr(e.id));
                }
            }
            hir::ExprBinary(ref op, ref lhs, ref rhs) |
            hir::ExprAssignOp(ref op, ref lhs, ref rhs) => {
                let lhs_ty = self.fcx.node_ty(lhs.id);
                let lhs_ty = self.fcx.resolve_type_vars_if_possible(&lhs_ty);

                let rhs_ty = self.fcx.node_ty(rhs.id);
                let rhs_ty = self.fcx.resolve_type_vars_if_possible(&rhs_ty);

                if lhs_ty.is_scalar() && rhs_ty.is_scalar() {
                    self.fcx.tables.borrow_mut().method_map.remove(&MethodCall::expr(e.id));

                    // weird but true: the by-ref binops put an
                    // adjustment on the lhs but not the rhs; the
                    // adjustment for rhs is kind of baked into the
                    // system.
                    match e.node {
                        hir::ExprBinary(..) => {
                            if !op.node.is_by_value() {
                                self.fcx.tables.borrow_mut().adjustments.remove(&lhs.id);
                            }
                        },
                        hir::ExprAssignOp(..) => {
                            self.fcx.tables.borrow_mut().adjustments.remove(&lhs.id);
                        },
                        _ => {},
                    }
                }
            }
            _ => {},
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Impl of Visitor for Resolver
//
// This is the master code which walks the AST. It delegates most of
// the heavy lifting to the generic visit and resolve functions
// below. In general, a function is made into a `visitor` if it must
// traffic in node-ids or update tables in the type context etc.

impl<'cx, 'gcx, 'tcx> Visitor<'gcx> for WritebackCx<'cx, 'gcx, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'gcx> {
        NestedVisitorMap::None
    }

    fn visit_stmt(&mut self, s: &'gcx hir::Stmt) {
        if self.fcx.writeback_errors.get() {
            return;
        }

        self.visit_node_id(ResolvingExpr(s.span), s.node.id());
        intravisit::walk_stmt(self, s);
    }

    fn visit_expr(&mut self, e: &'gcx hir::Expr) {
        if self.fcx.writeback_errors.get() {
            return;
        }

        self.fix_scalar_builtin_expr(e);

        self.visit_node_id(ResolvingExpr(e.span), e.id);
        self.visit_method_map_entry(ResolvingExpr(e.span),
                                    MethodCall::expr(e.id));

        if let hir::ExprClosure(_, _, body, _) = e.node {
            let body = self.fcx.tcx.map.body(body);
            for arg in &body.arguments {
                self.visit_node_id(ResolvingExpr(e.span), arg.id);
            }

            self.visit_body(body);
        }

        intravisit::walk_expr(self, e);
    }

    fn visit_block(&mut self, b: &'gcx hir::Block) {
        if self.fcx.writeback_errors.get() {
            return;
        }

        self.visit_node_id(ResolvingExpr(b.span), b.id);
        intravisit::walk_block(self, b);
    }

    fn visit_pat(&mut self, p: &'gcx hir::Pat) {
        if self.fcx.writeback_errors.get() {
            return;
        }

        self.visit_node_id(ResolvingPattern(p.span), p.id);

        intravisit::walk_pat(self, p);
    }

    fn visit_local(&mut self, l: &'gcx hir::Local) {
        if self.fcx.writeback_errors.get() {
            return;
        }

        let var_ty = self.fcx.local_ty(l.span, l.id);
        let var_ty = self.resolve(&var_ty, ResolvingLocal(l.span));
        self.write_ty_to_tables(l.id, var_ty);
        intravisit::walk_local(self, l);
    }
}

impl<'cx, 'gcx, 'tcx> WritebackCx<'cx, 'gcx, 'tcx> {
    fn visit_upvar_borrow_map(&mut self) {
        if self.fcx.writeback_errors.get() {
            return;
        }

        for (upvar_id, upvar_capture) in self.fcx.tables.borrow().upvar_capture_map.iter() {
            let new_upvar_capture = match *upvar_capture {
                ty::UpvarCapture::ByValue => ty::UpvarCapture::ByValue,
                ty::UpvarCapture::ByRef(ref upvar_borrow) => {
                    let r = upvar_borrow.region;
                    let r = self.resolve(&r, ResolvingUpvar(*upvar_id));
                    ty::UpvarCapture::ByRef(
                        ty::UpvarBorrow { kind: upvar_borrow.kind, region: r })
                }
            };
            debug!("Upvar capture for {:?} resolved to {:?}",
                   upvar_id,
                   new_upvar_capture);
            self.tables.upvar_capture_map.insert(*upvar_id, new_upvar_capture);
        }
    }

    fn visit_closures(&self) {
        if self.fcx.writeback_errors.get() {
            return
        }

        for (&id, closure_ty) in self.fcx.tables.borrow().closure_tys.iter() {
            let closure_ty = self.resolve(closure_ty, ResolvingClosure(id));
            let def_id = self.tcx().map.local_def_id(id);
            self.tcx().closure_tys.borrow_mut().insert(def_id, closure_ty);
        }

        for (&id, &closure_kind) in self.fcx.tables.borrow().closure_kinds.iter() {
            let def_id = self.tcx().map.local_def_id(id);
            self.tcx().closure_kinds.borrow_mut().insert(def_id, closure_kind);
        }
    }

    fn visit_anon_types(&self) {
        if self.fcx.writeback_errors.get() {
            return
        }

        let gcx = self.tcx().global_tcx();
        for (&def_id, &concrete_ty) in self.fcx.anon_types.borrow().iter() {
            let reason = ResolvingAnonTy(def_id);
            let inside_ty = self.resolve(&concrete_ty, reason);

            // Convert the type from the function into a type valid outside
            // the function, by replacing free regions with early-bound ones.
            let outside_ty = gcx.fold_regions(&inside_ty, &mut false, |r, _| {
                match *r {
                    // 'static is valid everywhere.
                    ty::ReStatic |
                    ty::ReEmpty => gcx.mk_region(*r),

                    // Free regions that come from early-bound regions are valid.
                    ty::ReFree(ty::FreeRegion {
                        bound_region: ty::BoundRegion::BrNamed(def_id, ..), ..
                    }) if self.free_to_bound_regions.contains_key(&def_id) => {
                        self.free_to_bound_regions[&def_id]
                    }

                    ty::ReFree(_) |
                    ty::ReEarlyBound(_) |
                    ty::ReLateBound(..) |
                    ty::ReScope(_) |
                    ty::ReSkolemized(..) => {
                        let span = reason.span(self.tcx());
                        span_err!(self.tcx().sess, span, E0564,
                                  "only named lifetimes are allowed in `impl Trait`, \
                                   but `{}` was found in the type `{}`", r, inside_ty);
                        gcx.mk_region(ty::ReStatic)
                    }

                    ty::ReVar(_) |
                    ty::ReErased => {
                        let span = reason.span(self.tcx());
                        span_bug!(span, "invalid region in impl Trait: {:?}", r);
                    }
                }
            });

            gcx.item_types.borrow_mut().insert(def_id, outside_ty);
        }
    }

    fn visit_node_id(&mut self, reason: ResolveReason, id: ast::NodeId) {
        // Export associated path extensions.
        if let Some(def) = self.fcx.tables.borrow_mut().type_relative_path_defs.remove(&id) {
            self.tables.type_relative_path_defs.insert(id, def);
        }

        // Resolve any borrowings for the node with id `id`
        self.visit_adjustments(reason, id);

        // Resolve the type of the node with id `id`
        let n_ty = self.fcx.node_ty(id);
        let n_ty = self.resolve(&n_ty, reason);
        self.write_ty_to_tables(id, n_ty);
        debug!("Node {} has type {:?}", id, n_ty);

        // Resolve any substitutions
        self.fcx.opt_node_ty_substs(id, |item_substs| {
            let item_substs = self.resolve(item_substs, reason);
            if !item_substs.is_noop() {
                debug!("write_substs_to_tcx({}, {:?})", id, item_substs);
                assert!(!item_substs.substs.needs_infer());
                self.tables.item_substs.insert(id, item_substs);
            }
        });
    }

    fn visit_adjustments(&mut self, reason: ResolveReason, id: ast::NodeId) {
        let adjustments = self.fcx.tables.borrow_mut().adjustments.remove(&id);
        match adjustments {
            None => {
                debug!("No adjustments for node {}", id);
            }

            Some(adjustment) => {
                let resolved_adjustment = match adjustment.kind {
                    adjustment::Adjust::NeverToAny => {
                        adjustment::Adjust::NeverToAny
                    }

                    adjustment::Adjust::ReifyFnPointer => {
                        adjustment::Adjust::ReifyFnPointer
                    }

                    adjustment::Adjust::MutToConstPointer => {
                        adjustment::Adjust::MutToConstPointer
                    }

                    adjustment::Adjust::UnsafeFnPointer => {
                        adjustment::Adjust::UnsafeFnPointer
                    }

                    adjustment::Adjust::DerefRef { autoderefs, autoref, unsize } => {
                        for autoderef in 0..autoderefs {
                            let method_call = MethodCall::autoderef(id, autoderef as u32);
                            self.visit_method_map_entry(reason, method_call);
                        }

                        adjustment::Adjust::DerefRef {
                            autoderefs: autoderefs,
                            autoref: self.resolve(&autoref, reason),
                            unsize: unsize,
                        }
                    }
                };
                let resolved_adjustment = adjustment::Adjustment {
                    kind: resolved_adjustment,
                    target: self.resolve(&adjustment.target, reason)
                };
                debug!("Adjustments for node {}: {:?}", id, resolved_adjustment);
                self.tables.adjustments.insert(id, resolved_adjustment);
            }
        }
    }

    fn visit_method_map_entry(&mut self,
                              reason: ResolveReason,
                              method_call: MethodCall) {
        // Resolve any method map entry
        let new_method = match self.fcx.tables.borrow_mut().method_map.remove(&method_call) {
            Some(method) => {
                debug!("writeback::resolve_method_map_entry(call={:?}, entry={:?})",
                       method_call,
                       method);
                let new_method = MethodCallee {
                    def_id: method.def_id,
                    ty: self.resolve(&method.ty, reason),
                    substs: self.resolve(&method.substs, reason),
                };

                Some(new_method)
            }
            None => None
        };

        //NB(jroesch): We need to match twice to avoid a double borrow which would cause an ICE
        if let Some(method) = new_method {
            self.tables.method_map.insert(method_call, method);
        }
    }

    fn visit_liberated_fn_sigs(&mut self) {
        for (&node_id, fn_sig) in self.fcx.tables.borrow().liberated_fn_sigs.iter() {
            let fn_sig = self.resolve(fn_sig, ResolvingFnSig(node_id));
            self.tables.liberated_fn_sigs.insert(node_id, fn_sig.clone());
        }
    }

    fn visit_fru_field_types(&mut self) {
        for (&node_id, ftys) in self.fcx.tables.borrow().fru_field_types.iter() {
            let ftys = self.resolve(ftys, ResolvingFieldTypes(node_id));
            self.tables.fru_field_types.insert(node_id, ftys);
        }
    }

    fn visit_deferred_obligations(&mut self, item_id: ast::NodeId) {
        let deferred_obligations = self.fcx.deferred_obligations.borrow();
        let obligations: Vec<_> = deferred_obligations.iter().map(|obligation| {
            let reason = ResolvingDeferredObligation(obligation.cause.span);
            self.resolve(obligation, reason)
        }).collect();

        if !obligations.is_empty() {
            assert!(self.fcx.ccx.deferred_obligations.borrow_mut()
                                .insert(item_id, obligations).is_none());
        }
    }

    fn visit_type_nodes(&self) {
        for (&id, ty) in self.fcx.ast_ty_to_ty_cache.borrow().iter() {
            let ty = self.resolve(ty, ResolvingTyNode(id));
            self.fcx.ccx.ast_ty_to_ty_cache.borrow_mut().insert(id, ty);
        }
    }

    fn resolve<T>(&self, x: &T, reason: ResolveReason) -> T::Lifted
        where T: TypeFoldable<'tcx> + ty::Lift<'gcx>
    {
        let x = x.fold_with(&mut Resolver::new(self.fcx, reason));
        if let Some(lifted) = self.tcx().lift_to_global(&x) {
            lifted
        } else {
            span_bug!(reason.span(self.tcx()),
                      "writeback: `{:?}` missing from the global type context", x);
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Resolution reason.

#[derive(Copy, Clone, Debug)]
enum ResolveReason {
    ResolvingExpr(Span),
    ResolvingLocal(Span),
    ResolvingPattern(Span),
    ResolvingUpvar(ty::UpvarId),
    ResolvingClosure(ast::NodeId),
    ResolvingFnSig(ast::NodeId),
    ResolvingFieldTypes(ast::NodeId),
    ResolvingAnonTy(DefId),
    ResolvingDeferredObligation(Span),
    ResolvingTyNode(ast::NodeId),
}

impl<'a, 'gcx, 'tcx> ResolveReason {
    fn span(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Span {
        match *self {
            ResolvingExpr(s) => s,
            ResolvingLocal(s) => s,
            ResolvingPattern(s) => s,
            ResolvingUpvar(upvar_id) => {
                tcx.expr_span(upvar_id.closure_expr_id)
            }
            ResolvingClosure(id) |
            ResolvingFnSig(id) |
            ResolvingFieldTypes(id) |
            ResolvingTyNode(id) => {
                tcx.map.span(id)
            }
            ResolvingAnonTy(did) => {
                tcx.def_span(did)
            }
            ResolvingDeferredObligation(span) => span
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// The Resolver. This is the type folding engine that detects
// unresolved types and so forth.

struct Resolver<'cx, 'gcx: 'cx+'tcx, 'tcx: 'cx> {
    tcx: TyCtxt<'cx, 'gcx, 'tcx>,
    infcx: &'cx InferCtxt<'cx, 'gcx, 'tcx>,
    writeback_errors: &'cx Cell<bool>,
    reason: ResolveReason,
}

impl<'cx, 'gcx, 'tcx> Resolver<'cx, 'gcx, 'tcx> {
    fn new(fcx: &'cx FnCtxt<'cx, 'gcx, 'tcx>,
           reason: ResolveReason)
           -> Resolver<'cx, 'gcx, 'tcx>
    {
        Resolver::from_infcx(fcx, &fcx.writeback_errors, reason)
    }

    fn from_infcx(infcx: &'cx InferCtxt<'cx, 'gcx, 'tcx>,
                  writeback_errors: &'cx Cell<bool>,
                  reason: ResolveReason)
                  -> Resolver<'cx, 'gcx, 'tcx>
    {
        Resolver { infcx: infcx,
                   tcx: infcx.tcx,
                   writeback_errors: writeback_errors,
                   reason: reason }
    }

    fn report_error(&self, e: FixupError) {
        self.writeback_errors.set(true);
        if !self.tcx.sess.has_errors() {
            match self.reason {
                ResolvingExpr(span) => {
                    struct_span_err!(
                        self.tcx.sess, span, E0101,
                        "cannot determine a type for this expression: {}", e)
                        .span_label(span, &format!("cannot resolve type of expression"))
                        .emit();
                }

                ResolvingLocal(span) => {
                    struct_span_err!(
                        self.tcx.sess, span, E0102,
                        "cannot determine a type for this local variable: {}", e)
                        .span_label(span, &format!("cannot resolve type of variable"))
                        .emit();
                }

                ResolvingPattern(span) => {
                    span_err!(self.tcx.sess, span, E0103,
                        "cannot determine a type for this pattern binding: {}", e);
                }

                ResolvingUpvar(upvar_id) => {
                    let span = self.reason.span(self.tcx);
                    span_err!(self.tcx.sess, span, E0104,
                        "cannot resolve lifetime for captured variable `{}`: {}",
                        self.tcx.local_var_name_str(upvar_id.var_id), e);
                }

                ResolvingClosure(_) => {
                    let span = self.reason.span(self.tcx);
                    span_err!(self.tcx.sess, span, E0196,
                              "cannot determine a type for this closure")
                }

                ResolvingFnSig(_) |
                ResolvingFieldTypes(_) |
                ResolvingDeferredObligation(_) |
                ResolvingTyNode(_) => {
                    // any failures here should also fail when
                    // resolving the patterns, closure types, or
                    // something else.
                    let span = self.reason.span(self.tcx);
                    self.tcx.sess.delay_span_bug(
                        span,
                        &format!("cannot resolve some aspect of data for {:?}: {}",
                                 self.reason, e));
                }

                ResolvingAnonTy(_) => {
                    let span = self.reason.span(self.tcx);
                    span_err!(self.tcx.sess, span, E0563,
                              "cannot determine a type for this `impl Trait`: {}", e)
                }
            }
        }
    }
}

impl<'cx, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for Resolver<'cx, 'gcx, 'tcx> {
    fn tcx<'a>(&'a self) -> TyCtxt<'a, 'gcx, 'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        match self.infcx.fully_resolve(&t) {
            Ok(t) => t,
            Err(e) => {
                debug!("Resolver::fold_ty: input type `{:?}` not fully resolvable",
                       t);
                self.report_error(e);
                self.tcx().types.err
            }
        }
    }

    fn fold_region(&mut self, r: &'tcx ty::Region) -> &'tcx ty::Region {
        match self.infcx.fully_resolve(&r) {
            Ok(r) => r,
            Err(e) => {
                self.report_error(e);
                self.tcx.mk_region(ty::ReStatic)
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// During type check, we store promises with the result of trait
// lookup rather than the actual results (because the results are not
// necessarily available immediately). These routines unwind the
// promises. It is expected that we will have already reported any
// errors that may be encountered, so if the promises store an error,
// a dummy result is returned.
