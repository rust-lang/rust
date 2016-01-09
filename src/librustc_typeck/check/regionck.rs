// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The region check is a final pass that runs over the AST after we have
//! inferred the type constraints but before we have actually finalized
//! the types.  Its purpose is to embed a variety of region constraints.
//! Inserting these constraints as a separate pass is good because (1) it
//! localizes the code that has to do with region inference and (2) often
//! we cannot know what constraints are needed until the basic types have
//! been inferred.
//!
//! ### Interaction with the borrow checker
//!
//! In general, the job of the borrowck module (which runs later) is to
//! check that all soundness criteria are met, given a particular set of
//! regions. The job of *this* module is to anticipate the needs of the
//! borrow checker and infer regions that will satisfy its requirements.
//! It is generally true that the inference doesn't need to be sound,
//! meaning that if there is a bug and we inferred bad regions, the borrow
//! checker should catch it. This is not entirely true though; for
//! example, the borrow checker doesn't check subtyping, and it doesn't
//! check that region pointers are always live when they are used. It
//! might be worthwhile to fix this so that borrowck serves as a kind of
//! verification step -- that would add confidence in the overall
//! correctness of the compiler, at the cost of duplicating some type
//! checks and effort.
//!
//! ### Inferring the duration of borrows, automatic and otherwise
//!
//! Whenever we introduce a borrowed pointer, for example as the result of
//! a borrow expression `let x = &data`, the lifetime of the pointer `x`
//! is always specified as a region inference variable. `regionck` has the
//! job of adding constraints such that this inference variable is as
//! narrow as possible while still accommodating all uses (that is, every
//! dereference of the resulting pointer must be within the lifetime).
//!
//! #### Reborrows
//!
//! Generally speaking, `regionck` does NOT try to ensure that the data
//! `data` will outlive the pointer `x`. That is the job of borrowck.  The
//! one exception is when "re-borrowing" the contents of another borrowed
//! pointer. For example, imagine you have a borrowed pointer `b` with
//! lifetime L1 and you have an expression `&*b`. The result of this
//! expression will be another borrowed pointer with lifetime L2 (which is
//! an inference variable). The borrow checker is going to enforce the
//! constraint that L2 < L1, because otherwise you are re-borrowing data
//! for a lifetime larger than the original loan.  However, without the
//! routines in this module, the region inferencer would not know of this
//! dependency and thus it might infer the lifetime of L2 to be greater
//! than L1 (issue #3148).
//!
//! There are a number of troublesome scenarios in the tests
//! `region-dependent-*.rs`, but here is one example:
//!
//!     struct Foo { i: i32 }
//!     struct Bar { foo: Foo  }
//!     fn get_i<'a>(x: &'a Bar) -> &'a i32 {
//!        let foo = &x.foo; // Lifetime L1
//!        &foo.i            // Lifetime L2
//!     }
//!
//! Note that this comes up either with `&` expressions, `ref`
//! bindings, and `autorefs`, which are the three ways to introduce
//! a borrow.
//!
//! The key point here is that when you are borrowing a value that
//! is "guaranteed" by a borrowed pointer, you must link the
//! lifetime of that borrowed pointer (L1, here) to the lifetime of
//! the borrow itself (L2).  What do I mean by "guaranteed" by a
//! borrowed pointer? I mean any data that is reached by first
//! dereferencing a borrowed pointer and then either traversing
//! interior offsets or boxes.  We say that the guarantor
//! of such data is the region of the borrowed pointer that was
//! traversed.  This is essentially the same as the ownership
//! relation, except that a borrowed pointer never owns its
//! contents.

use astconv::AstConv;
use check::dropck;
use check::FnCtxt;
use middle::free_region::FreeRegionMap;
use middle::implicator::{self, Implication};
use middle::mem_categorization as mc;
use middle::mem_categorization::Categorization;
use middle::region::{self, CodeExtent};
use middle::subst::Substs;
use middle::traits;
use middle::ty::{self, Ty, MethodCall, TypeFoldable};
use middle::infer::{self, GenericKind, InferCtxt, SubregionOrigin, TypeOrigin, VerifyBound};
use middle::pat_util;
use middle::ty::adjustment;
use middle::ty::wf::ImpliedBound;

use std::mem;
use syntax::ast;
use syntax::codemap::Span;
use rustc_front::intravisit::{self, Visitor};
use rustc_front::hir;
use rustc_front::util as hir_util;

use self::SubjectNode::Subject;

// a variation on try that just returns unit
macro_rules! ignore_err {
    ($e:expr) => (match $e { Ok(e) => e, Err(_) => return () })
}

///////////////////////////////////////////////////////////////////////////
// PUBLIC ENTRY POINTS

pub fn regionck_expr(fcx: &FnCtxt, e: &hir::Expr) {
    let mut rcx = Rcx::new(fcx, RepeatingScope(e.id), e.id, Subject(e.id));
    if fcx.err_count_since_creation() == 0 {
        // regionck assumes typeck succeeded
        rcx.visit_expr(e);
        rcx.visit_region_obligations(e.id);
    }
    rcx.resolve_regions_and_report_errors();
}

/// Region checking during the WF phase for items. `wf_tys` are the
/// types from which we should derive implied bounds, if any.
pub fn regionck_item<'a,'tcx>(fcx: &FnCtxt<'a,'tcx>,
                              item_id: ast::NodeId,
                              span: Span,
                              wf_tys: &[Ty<'tcx>]) {
    debug!("regionck_item(item.id={:?}, wf_tys={:?}", item_id, wf_tys);
    let mut rcx = Rcx::new(fcx, RepeatingScope(item_id), item_id, Subject(item_id));
    let tcx = fcx.tcx();
    rcx.free_region_map
       .relate_free_regions_from_predicates(tcx, &fcx.infcx().parameter_environment.caller_bounds);
    rcx.relate_free_regions(wf_tys, item_id, span);
    rcx.visit_region_obligations(item_id);
    rcx.resolve_regions_and_report_errors();
}

pub fn regionck_fn(fcx: &FnCtxt,
                   fn_id: ast::NodeId,
                   fn_span: Span,
                   decl: &hir::FnDecl,
                   blk: &hir::Block) {
    debug!("regionck_fn(id={})", fn_id);
    let mut rcx = Rcx::new(fcx, RepeatingScope(blk.id), blk.id, Subject(fn_id));

    if fcx.err_count_since_creation() == 0 {
        // regionck assumes typeck succeeded
        rcx.visit_fn_body(fn_id, decl, blk, fn_span);
    }

    let tcx = fcx.tcx();
    rcx.free_region_map
       .relate_free_regions_from_predicates(tcx, &fcx.infcx().parameter_environment.caller_bounds);

    rcx.resolve_regions_and_report_errors();

    // For the top-level fn, store the free-region-map. We don't store
    // any map for closures; they just share the same map as the
    // function that created them.
    fcx.tcx().store_free_region_map(fn_id, rcx.free_region_map);
}

///////////////////////////////////////////////////////////////////////////
// INTERNALS

pub struct Rcx<'a, 'tcx: 'a> {
    pub fcx: &'a FnCtxt<'a, 'tcx>,

    region_bound_pairs: Vec<(ty::Region, GenericKind<'tcx>)>,

    free_region_map: FreeRegionMap,

    // id of innermost fn body id
    body_id: ast::NodeId,

    // call_site scope of innermost fn
    call_site_scope: Option<CodeExtent>,

    // id of innermost fn or loop
    repeating_scope: ast::NodeId,

    // id of AST node being analyzed (the subject of the analysis).
    subject: SubjectNode,

}

pub struct RepeatingScope(ast::NodeId);
pub enum SubjectNode { Subject(ast::NodeId), None }

impl<'a, 'tcx> Rcx<'a, 'tcx> {
    pub fn new(fcx: &'a FnCtxt<'a, 'tcx>,
               initial_repeating_scope: RepeatingScope,
               initial_body_id: ast::NodeId,
               subject: SubjectNode) -> Rcx<'a, 'tcx> {
        let RepeatingScope(initial_repeating_scope) = initial_repeating_scope;
        Rcx { fcx: fcx,
              repeating_scope: initial_repeating_scope,
              body_id: initial_body_id,
              call_site_scope: None,
              subject: subject,
              region_bound_pairs: Vec::new(),
              free_region_map: FreeRegionMap::new(),
        }
    }

    pub fn tcx(&self) -> &'a ty::ctxt<'tcx> {
        self.fcx.ccx.tcx
    }

    pub fn infcx(&self) -> &InferCtxt<'a,'tcx> {
        self.fcx.infcx()
    }

    fn set_call_site_scope(&mut self, call_site_scope: Option<CodeExtent>) -> Option<CodeExtent> {
        mem::replace(&mut self.call_site_scope, call_site_scope)
    }

    fn set_body_id(&mut self, body_id: ast::NodeId) -> ast::NodeId {
        mem::replace(&mut self.body_id, body_id)
    }

    fn set_repeating_scope(&mut self, scope: ast::NodeId) -> ast::NodeId {
        mem::replace(&mut self.repeating_scope, scope)
    }

    /// Try to resolve the type for the given node, returning t_err if an error results.  Note that
    /// we never care about the details of the error, the same error will be detected and reported
    /// in the writeback phase.
    ///
    /// Note one important point: we do not attempt to resolve *region variables* here.  This is
    /// because regionck is essentially adding constraints to those region variables and so may yet
    /// influence how they are resolved.
    ///
    /// Consider this silly example:
    ///
    /// ```
    /// fn borrow(x: &i32) -> &i32 {x}
    /// fn foo(x: @i32) -> i32 {  // block: B
    ///     let b = borrow(x);    // region: <R0>
    ///     *b
    /// }
    /// ```
    ///
    /// Here, the region of `b` will be `<R0>`.  `<R0>` is constrained to be some subregion of the
    /// block B and some superregion of the call.  If we forced it now, we'd choose the smaller
    /// region (the call).  But that would make the *b illegal.  Since we don't resolve, the type
    /// of b will be `&<R0>.i32` and then `*b` will require that `<R0>` be bigger than the let and
    /// the `*b` expression, so we will effectively resolve `<R0>` to be the block B.
    pub fn resolve_type(&self, unresolved_ty: Ty<'tcx>) -> Ty<'tcx> {
        self.fcx.infcx().resolve_type_vars_if_possible(&unresolved_ty)
    }

    /// Try to resolve the type for the given node.
    fn resolve_node_type(&self, id: ast::NodeId) -> Ty<'tcx> {
        let t = self.fcx.node_ty(id);
        self.resolve_type(t)
    }

    fn resolve_method_type(&self, method_call: MethodCall) -> Option<Ty<'tcx>> {
        let method_ty = self.fcx.inh.tables.borrow().method_map
                            .get(&method_call).map(|method| method.ty);
        method_ty.map(|method_ty| self.resolve_type(method_ty))
    }

    /// Try to resolve the type for the given node.
    pub fn resolve_expr_type_adjusted(&mut self, expr: &hir::Expr) -> Ty<'tcx> {
        let ty_unadjusted = self.resolve_node_type(expr.id);
        if ty_unadjusted.references_error() {
            ty_unadjusted
        } else {
            ty_unadjusted.adjust(
                self.fcx.tcx(), expr.span, expr.id,
                self.fcx.inh.tables.borrow().adjustments.get(&expr.id),
                |method_call| self.resolve_method_type(method_call))
        }
    }

    fn visit_fn_body(&mut self,
                     id: ast::NodeId, // the id of the fn itself
                     fn_decl: &hir::FnDecl,
                     body: &hir::Block,
                     span: Span)
    {
        // When we enter a function, we can derive
        debug!("visit_fn_body(id={})", id);

        let call_site = self.fcx.tcx().region_maps.lookup_code_extent(
            region::CodeExtentData::CallSiteScope { fn_id: id, body_id: body.id });
        let old_call_site_scope = self.set_call_site_scope(Some(call_site));

        let fn_sig = {
            let fn_sig_map = &self.infcx().tables.borrow().liberated_fn_sigs;
            match fn_sig_map.get(&id) {
                Some(f) => f.clone(),
                None => {
                    self.tcx().sess.bug(
                        &format!("No fn-sig entry for id={}", id));
                }
            }
        };

        let old_region_bounds_pairs_len = self.region_bound_pairs.len();

        // Collect the types from which we create inferred bounds.
        // For the return type, if diverging, substitute `bool` just
        // because it will have no effect.
        //
        // FIXME(#27579) return types should not be implied bounds
        let fn_sig_tys: Vec<_> =
            fn_sig.inputs.iter()
                         .cloned()
                         .chain(Some(fn_sig.output.unwrap_or(self.tcx().types.bool)))
                         .collect();

        let old_body_id = self.set_body_id(body.id);
        self.relate_free_regions(&fn_sig_tys[..], body.id, span);
        link_fn_args(self,
                     self.tcx().region_maps.node_extent(body.id),
                     &fn_decl.inputs[..]);
        self.visit_block(body);
        self.visit_region_obligations(body.id);

        let call_site_scope = self.call_site_scope.unwrap();
        debug!("visit_fn_body body.id {} call_site_scope: {:?}",
               body.id, call_site_scope);
        type_of_node_must_outlive(self,
                                  infer::CallReturn(span),
                                  body.id,
                                  ty::ReScope(call_site_scope));

        self.region_bound_pairs.truncate(old_region_bounds_pairs_len);

        self.set_body_id(old_body_id);
        self.set_call_site_scope(old_call_site_scope);
    }

    fn visit_region_obligations(&mut self, node_id: ast::NodeId)
    {
        debug!("visit_region_obligations: node_id={}", node_id);

        // region checking can introduce new pending obligations
        // which, when processed, might generate new region
        // obligations. So make sure we process those.
        self.fcx.select_all_obligations_or_error();

        // Make a copy of the region obligations vec because we'll need
        // to be able to borrow the fulfillment-cx below when projecting.
        let region_obligations =
            self.fcx
                .inh
                .infcx
                .fulfillment_cx
                .borrow()
                .region_obligations(node_id)
                .to_vec();

        for r_o in &region_obligations {
            debug!("visit_region_obligations: r_o={:?} cause={:?}",
                   r_o, r_o.cause);
            let sup_type = self.resolve_type(r_o.sup_type);
            let origin = self.code_to_origin(r_o.cause.span, sup_type, &r_o.cause.code);

            if r_o.sub_region != ty::ReEmpty {
                type_must_outlive(self, origin, sup_type, r_o.sub_region);
            } else {
                self.visit_old_school_wf(node_id, sup_type, origin);
            }
        }

        // Processing the region obligations should not cause the list to grow further:
        assert_eq!(region_obligations.len(),
                   self.fcx.inh.infcx.fulfillment_cx.borrow().region_obligations(node_id).len());
    }

    fn visit_old_school_wf(&mut self,
                           body_id: ast::NodeId,
                           ty: Ty<'tcx>,
                           origin: infer::SubregionOrigin<'tcx>) {
        // As a weird kind of hack, we use a region of empty as a signal
        // to mean "old-school WF rules". The only reason the old-school
        // WF rules are not encoded using WF is that this leads to errors,
        // and we want to phase those in gradually.

        // FIXME(#27579) remove this weird special case once we phase in new WF rules completely
        let implications = implicator::implications(self.infcx(),
                                                    body_id,
                                                    ty,
                                                    ty::ReEmpty,
                                                    origin.span());
        let origin_for_ty = |ty: Option<Ty<'tcx>>| match ty {
            None => origin.clone(),
            Some(ty) => infer::ReferenceOutlivesReferent(ty, origin.span()),
        };
        for implication in implications {
            match implication {
                Implication::RegionSubRegion(ty, r1, r2) => {
                    self.fcx.mk_subr(origin_for_ty(ty), r1, r2);
                }
                Implication::RegionSubGeneric(ty, r1, GenericKind::Param(param_ty)) => {
                    param_ty_must_outlive(self, origin_for_ty(ty), r1, param_ty);
                }
                Implication::RegionSubGeneric(ty, r1, GenericKind::Projection(proj_ty)) => {
                    projection_must_outlive(self, origin_for_ty(ty), r1, proj_ty);
                }
                Implication::Predicate(def_id, predicate) => {
                    let cause = traits::ObligationCause::new(origin.span(),
                                                             body_id,
                                                             traits::ItemObligation(def_id));
                    let obligation = traits::Obligation::new(cause, predicate);
                    self.fcx.register_predicate(obligation);
                }
            }
        }
    }

    fn code_to_origin(&self,
                      span: Span,
                      sup_type: Ty<'tcx>,
                      code: &traits::ObligationCauseCode<'tcx>)
                      -> SubregionOrigin<'tcx> {
        match *code {
            traits::ObligationCauseCode::ReferenceOutlivesReferent(ref_type) =>
                infer::ReferenceOutlivesReferent(ref_type, span),
            _ =>
                infer::RelateParamBound(span, sup_type),
        }
    }

    /// This method populates the region map's `free_region_map`. It walks over the transformed
    /// argument and return types for each function just before we check the body of that function,
    /// looking for types where you have a borrowed pointer to other borrowed data (e.g., `&'a &'b
    /// [usize]`.  We do not allow references to outlive the things they point at, so we can assume
    /// that `'a <= 'b`. This holds for both the argument and return types, basically because, on
    /// the caller side, the caller is responsible for checking that the type of every expression
    /// (including the actual values for the arguments, as well as the return type of the fn call)
    /// is well-formed.
    ///
    /// Tests: `src/test/compile-fail/regions-free-region-ordering-*.rs`
    fn relate_free_regions(&mut self,
                           fn_sig_tys: &[Ty<'tcx>],
                           body_id: ast::NodeId,
                           span: Span) {
        debug!("relate_free_regions >>");

        for &ty in fn_sig_tys {
            let ty = self.resolve_type(ty);
            debug!("relate_free_regions(t={:?})", ty);
            let implied_bounds = ty::wf::implied_bounds(self.fcx.infcx(), body_id, ty, span);

            // Record any relations between free regions that we observe into the free-region-map.
            self.free_region_map.relate_free_regions_from_implied_bounds(&implied_bounds);

            // But also record other relationships, such as `T:'x`,
            // that don't go into the free-region-map but which we use
            // here.
            for implication in implied_bounds {
                debug!("implication: {:?}", implication);
                match implication {
                    ImpliedBound::RegionSubRegion(ty::ReFree(free_a),
                                                  ty::ReVar(vid_b)) => {
                        self.fcx.inh.infcx.add_given(free_a, vid_b);
                    }
                    ImpliedBound::RegionSubParam(r_a, param_b) => {
                        self.region_bound_pairs.push((r_a, GenericKind::Param(param_b)));
                    }
                    ImpliedBound::RegionSubProjection(r_a, projection_b) => {
                        self.region_bound_pairs.push((r_a, GenericKind::Projection(projection_b)));
                    }
                    ImpliedBound::RegionSubRegion(..) => {
                        // In principle, we could record (and take
                        // advantage of) every relationship here, but
                        // we are also free not to -- it simply means
                        // strictly less that we can successfully type
                        // check. (It may also be that we should
                        // revise our inference system to be more
                        // general and to make use of *every*
                        // relationship that arises here, but
                        // presently we do not.)
                    }
                }
            }
        }

        debug!("<< relate_free_regions");
    }

    fn resolve_regions_and_report_errors(&self) {
        let subject_node_id = match self.subject {
            Subject(s) => s,
            SubjectNode::None => {
                self.tcx().sess.bug("cannot resolve_regions_and_report_errors \
                                     without subject node");
            }
        };

        self.fcx.infcx().resolve_regions_and_report_errors(&self.free_region_map,
                                                           subject_node_id);
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for Rcx<'a, 'tcx> {
    // (..) FIXME(#3238) should use visit_pat, not visit_arm/visit_local,
    // However, right now we run into an issue whereby some free
    // regions are not properly related if they appear within the
    // types of arguments that must be inferred. This could be
    // addressed by deferring the construction of the region
    // hierarchy, and in particular the relationships between free
    // regions, until regionck, as described in #3238.

    fn visit_fn(&mut self, _fk: intravisit::FnKind<'v>, fd: &'v hir::FnDecl,
                b: &'v hir::Block, span: Span, id: ast::NodeId) {
        self.visit_fn_body(id, fd, b, span)
    }

    fn visit_expr(&mut self, ex: &hir::Expr) { visit_expr(self, ex); }

    //visit_pat: visit_pat, // (..) see above

    fn visit_arm(&mut self, a: &hir::Arm) { visit_arm(self, a); }

    fn visit_local(&mut self, l: &hir::Local) { visit_local(self, l); }

    fn visit_block(&mut self, b: &hir::Block) { visit_block(self, b); }
}

fn visit_block(rcx: &mut Rcx, b: &hir::Block) {
    intravisit::walk_block(rcx, b);
}

fn visit_arm(rcx: &mut Rcx, arm: &hir::Arm) {
    // see above
    for p in &arm.pats {
        constrain_bindings_in_pat(&**p, rcx);
    }

    intravisit::walk_arm(rcx, arm);
}

fn visit_local(rcx: &mut Rcx, l: &hir::Local) {
    // see above
    constrain_bindings_in_pat(&*l.pat, rcx);
    link_local(rcx, l);
    intravisit::walk_local(rcx, l);
}

fn constrain_bindings_in_pat(pat: &hir::Pat, rcx: &mut Rcx) {
    let tcx = rcx.fcx.tcx();
    debug!("regionck::visit_pat(pat={:?})", pat);
    pat_util::pat_bindings(&tcx.def_map, pat, |_, id, span, _| {
        // If we have a variable that contains region'd data, that
        // data will be accessible from anywhere that the variable is
        // accessed. We must be wary of loops like this:
        //
        //     // from src/test/compile-fail/borrowck-lend-flow.rs
        //     let mut v = box 3, w = box 4;
        //     let mut x = &mut w;
        //     loop {
        //         **x += 1;   // (2)
        //         borrow(v);  //~ ERROR cannot borrow
        //         x = &mut v; // (1)
        //     }
        //
        // Typically, we try to determine the region of a borrow from
        // those points where it is dereferenced. In this case, one
        // might imagine that the lifetime of `x` need only be the
        // body of the loop. But of course this is incorrect because
        // the pointer that is created at point (1) is consumed at
        // point (2), meaning that it must be live across the loop
        // iteration. The easiest way to guarantee this is to require
        // that the lifetime of any regions that appear in a
        // variable's type enclose at least the variable's scope.

        let var_scope = tcx.region_maps.var_scope(id);

        let origin = infer::BindingTypeIsNotValidAtDecl(span);
        type_of_node_must_outlive(rcx, origin, id, ty::ReScope(var_scope));

        let typ = rcx.resolve_node_type(id);
        dropck::check_safety_of_destructor_if_necessary(rcx, typ, span, var_scope);
    })
}

fn visit_expr(rcx: &mut Rcx, expr: &hir::Expr) {
    debug!("regionck::visit_expr(e={:?}, repeating_scope={})",
           expr, rcx.repeating_scope);

    // No matter what, the type of each expression must outlive the
    // scope of that expression. This also guarantees basic WF.
    let expr_ty = rcx.resolve_node_type(expr.id);
    // the region corresponding to this expression
    let expr_region = ty::ReScope(rcx.tcx().region_maps.node_extent(expr.id));
    type_must_outlive(rcx, infer::ExprTypeIsNotInScope(expr_ty, expr.span),
                      expr_ty, expr_region);

    let method_call = MethodCall::expr(expr.id);
    let opt_method_callee = rcx.fcx.inh.tables.borrow().method_map.get(&method_call).cloned();
    let has_method_map = opt_method_callee.is_some();

    // If we are calling a method (either explicitly or via an
    // overloaded operator), check that all of the types provided as
    // arguments for its type parameters are well-formed, and all the regions
    // provided as arguments outlive the call.
    if let Some(callee) = opt_method_callee {
        let origin = match expr.node {
            hir::ExprMethodCall(..) =>
                infer::ParameterOrigin::MethodCall,
            hir::ExprUnary(op, _) if op == hir::UnDeref =>
                infer::ParameterOrigin::OverloadedDeref,
            _ =>
                infer::ParameterOrigin::OverloadedOperator
        };

        substs_wf_in_scope(rcx, origin, &callee.substs, expr.span, expr_region);
        type_must_outlive(rcx, infer::ExprTypeIsNotInScope(callee.ty, expr.span),
                          callee.ty, expr_region);
    }

    // Check any autoderefs or autorefs that appear.
    let adjustment = rcx.fcx.inh.tables.borrow().adjustments.get(&expr.id).map(|a| a.clone());
    if let Some(adjustment) = adjustment {
        debug!("adjustment={:?}", adjustment);
        match adjustment {
            adjustment::AdjustDerefRef(adjustment::AutoDerefRef {
                autoderefs, ref autoref, ..
            }) => {
                let expr_ty = rcx.resolve_node_type(expr.id);
                constrain_autoderefs(rcx, expr, autoderefs, expr_ty);
                if let Some(ref autoref) = *autoref {
                    link_autoref(rcx, expr, autoderefs, autoref);

                    // Require that the resulting region encompasses
                    // the current node.
                    //
                    // FIXME(#6268) remove to support nested method calls
                    type_of_node_must_outlive(
                        rcx, infer::AutoBorrow(expr.span),
                        expr.id, expr_region);
                }
            }
            /*
            adjustment::AutoObject(_, ref bounds, _, _) => {
                // Determine if we are casting `expr` to a trait
                // instance. If so, we have to be sure that the type
                // of the source obeys the new region bound.
                let source_ty = rcx.resolve_node_type(expr.id);
                type_must_outlive(rcx, infer::RelateObjectBound(expr.span),
                                  source_ty, bounds.region_bound);
            }
            */
            _ => {}
        }

        // If necessary, constrain destructors in the unadjusted form of this
        // expression.
        let cmt_result = {
            let mc = mc::MemCategorizationContext::new(rcx.fcx.infcx());
            mc.cat_expr_unadjusted(expr)
        };
        match cmt_result {
            Ok(head_cmt) => {
                check_safety_of_rvalue_destructor_if_necessary(rcx,
                                                               head_cmt,
                                                               expr.span);
            }
            Err(..) => {
                let tcx = rcx.fcx.tcx();
                tcx.sess.delay_span_bug(expr.span, "cat_expr_unadjusted Errd");
            }
        }
    }

    // If necessary, constrain destructors in this expression. This will be
    // the adjusted form if there is an adjustment.
    let cmt_result = {
        let mc = mc::MemCategorizationContext::new(rcx.fcx.infcx());
        mc.cat_expr(expr)
    };
    match cmt_result {
        Ok(head_cmt) => {
            check_safety_of_rvalue_destructor_if_necessary(rcx, head_cmt, expr.span);
        }
        Err(..) => {
            let tcx = rcx.fcx.tcx();
            tcx.sess.delay_span_bug(expr.span, "cat_expr Errd");
        }
    }

    debug!("regionck::visit_expr(e={:?}, repeating_scope={}) - visiting subexprs",
           expr, rcx.repeating_scope);
    match expr.node {
        hir::ExprPath(..) => {
            rcx.fcx.opt_node_ty_substs(expr.id, |item_substs| {
                let origin = infer::ParameterOrigin::Path;
                substs_wf_in_scope(rcx, origin, &item_substs.substs, expr.span, expr_region);
            });
        }

        hir::ExprCall(ref callee, ref args) => {
            if has_method_map {
                constrain_call(rcx, expr, Some(&**callee),
                               args.iter().map(|e| &**e), false);
            } else {
                constrain_callee(rcx, callee.id, expr, &**callee);
                constrain_call(rcx, expr, None,
                               args.iter().map(|e| &**e), false);
            }

            intravisit::walk_expr(rcx, expr);
        }

        hir::ExprMethodCall(_, _, ref args) => {
            constrain_call(rcx, expr, Some(&*args[0]),
                           args[1..].iter().map(|e| &**e), false);

            intravisit::walk_expr(rcx, expr);
        }

        hir::ExprAssignOp(_, ref lhs, ref rhs) => {
            if has_method_map {
                constrain_call(rcx, expr, Some(&**lhs),
                               Some(&**rhs).into_iter(), false);
            }

            intravisit::walk_expr(rcx, expr);
        }

        hir::ExprIndex(ref lhs, ref rhs) if has_method_map => {
            constrain_call(rcx, expr, Some(&**lhs),
                           Some(&**rhs).into_iter(), true);

            intravisit::walk_expr(rcx, expr);
        },

        hir::ExprBinary(op, ref lhs, ref rhs) if has_method_map => {
            let implicitly_ref_args = !hir_util::is_by_value_binop(op.node);

            // As `expr_method_call`, but the call is via an
            // overloaded op.  Note that we (sadly) currently use an
            // implicit "by ref" sort of passing style here.  This
            // should be converted to an adjustment!
            constrain_call(rcx, expr, Some(&**lhs),
                           Some(&**rhs).into_iter(), implicitly_ref_args);

            intravisit::walk_expr(rcx, expr);
        }

        hir::ExprBinary(_, ref lhs, ref rhs) => {
            // If you do `x OP y`, then the types of `x` and `y` must
            // outlive the operation you are performing.
            let lhs_ty = rcx.resolve_expr_type_adjusted(&**lhs);
            let rhs_ty = rcx.resolve_expr_type_adjusted(&**rhs);
            for &ty in &[lhs_ty, rhs_ty] {
                type_must_outlive(rcx,
                                  infer::Operand(expr.span),
                                  ty,
                                  expr_region);
            }
            intravisit::walk_expr(rcx, expr);
        }

        hir::ExprUnary(op, ref lhs) if has_method_map => {
            let implicitly_ref_args = !hir_util::is_by_value_unop(op);

            // As above.
            constrain_call(rcx, expr, Some(&**lhs),
                           None::<hir::Expr>.iter(), implicitly_ref_args);

            intravisit::walk_expr(rcx, expr);
        }

        hir::ExprUnary(hir::UnDeref, ref base) => {
            // For *a, the lifetime of a must enclose the deref
            let method_call = MethodCall::expr(expr.id);
            let base_ty = match rcx.fcx.inh.tables.borrow().method_map.get(&method_call) {
                Some(method) => {
                    constrain_call(rcx, expr, Some(&**base),
                                   None::<hir::Expr>.iter(), true);
                    let fn_ret = // late-bound regions in overloaded method calls are instantiated
                        rcx.tcx().no_late_bound_regions(&method.ty.fn_ret()).unwrap();
                    fn_ret.unwrap()
                }
                None => rcx.resolve_node_type(base.id)
            };
            if let ty::TyRef(r_ptr, _) = base_ty.sty {
                mk_subregion_due_to_dereference(
                    rcx, expr.span, expr_region, *r_ptr);
            }

            intravisit::walk_expr(rcx, expr);
        }

        hir::ExprIndex(ref vec_expr, _) => {
            // For a[b], the lifetime of a must enclose the deref
            let vec_type = rcx.resolve_expr_type_adjusted(&**vec_expr);
            constrain_index(rcx, expr, vec_type);

            intravisit::walk_expr(rcx, expr);
        }

        hir::ExprCast(ref source, _) => {
            // Determine if we are casting `source` to a trait
            // instance.  If so, we have to be sure that the type of
            // the source obeys the trait's region bound.
            constrain_cast(rcx, expr, &**source);
            intravisit::walk_expr(rcx, expr);
        }

        hir::ExprAddrOf(m, ref base) => {
            link_addr_of(rcx, expr, m, &**base);

            // Require that when you write a `&expr` expression, the
            // resulting pointer has a lifetime that encompasses the
            // `&expr` expression itself. Note that we constraining
            // the type of the node expr.id here *before applying
            // adjustments*.
            //
            // FIXME(#6268) nested method calls requires that this rule change
            let ty0 = rcx.resolve_node_type(expr.id);
            type_must_outlive(rcx, infer::AddrOf(expr.span), ty0, expr_region);
            intravisit::walk_expr(rcx, expr);
        }

        hir::ExprMatch(ref discr, ref arms, _) => {
            link_match(rcx, &**discr, &arms[..]);

            intravisit::walk_expr(rcx, expr);
        }

        hir::ExprClosure(_, _, ref body) => {
            check_expr_fn_block(rcx, expr, &**body);
        }

        hir::ExprLoop(ref body, _) => {
            let repeating_scope = rcx.set_repeating_scope(body.id);
            intravisit::walk_expr(rcx, expr);
            rcx.set_repeating_scope(repeating_scope);
        }

        hir::ExprWhile(ref cond, ref body, _) => {
            let repeating_scope = rcx.set_repeating_scope(cond.id);
            rcx.visit_expr(&**cond);

            rcx.set_repeating_scope(body.id);
            rcx.visit_block(&**body);

            rcx.set_repeating_scope(repeating_scope);
        }

        hir::ExprRet(Some(ref ret_expr)) => {
            let call_site_scope = rcx.call_site_scope;
            debug!("visit_expr ExprRet ret_expr.id {} call_site_scope: {:?}",
                   ret_expr.id, call_site_scope);
            type_of_node_must_outlive(rcx,
                                      infer::CallReturn(ret_expr.span),
                                      ret_expr.id,
                                      ty::ReScope(call_site_scope.unwrap()));
            intravisit::walk_expr(rcx, expr);
        }

        _ => {
            intravisit::walk_expr(rcx, expr);
        }
    }
}

fn constrain_cast(rcx: &mut Rcx,
                  cast_expr: &hir::Expr,
                  source_expr: &hir::Expr)
{
    debug!("constrain_cast(cast_expr={:?}, source_expr={:?})",
           cast_expr,
           source_expr);

    let source_ty = rcx.resolve_node_type(source_expr.id);
    let target_ty = rcx.resolve_node_type(cast_expr.id);

    walk_cast(rcx, cast_expr, source_ty, target_ty);

    fn walk_cast<'a, 'tcx>(rcx: &mut Rcx<'a, 'tcx>,
                           cast_expr: &hir::Expr,
                           from_ty: Ty<'tcx>,
                           to_ty: Ty<'tcx>) {
        debug!("walk_cast(from_ty={:?}, to_ty={:?})",
               from_ty,
               to_ty);
        match (&from_ty.sty, &to_ty.sty) {
            /*From:*/ (&ty::TyRef(from_r, ref from_mt),
            /*To:  */  &ty::TyRef(to_r, ref to_mt)) => {
                // Target cannot outlive source, naturally.
                rcx.fcx.mk_subr(infer::Reborrow(cast_expr.span), *to_r, *from_r);
                walk_cast(rcx, cast_expr, from_mt.ty, to_mt.ty);
            }

            /*From:*/ (_,
            /*To:  */  &ty::TyTrait(box ty::TraitTy { ref bounds, .. })) => {
                // When T is existentially quantified as a trait
                // `Foo+'to`, it must outlive the region bound `'to`.
                type_must_outlive(rcx, infer::RelateObjectBound(cast_expr.span),
                                  from_ty, bounds.region_bound);
            }

            /*From:*/ (&ty::TyBox(from_referent_ty),
            /*To:  */  &ty::TyBox(to_referent_ty)) => {
                walk_cast(rcx, cast_expr, from_referent_ty, to_referent_ty);
            }

            _ => { }
        }
    }
}

fn check_expr_fn_block(rcx: &mut Rcx,
                       expr: &hir::Expr,
                       body: &hir::Block) {
    let repeating_scope = rcx.set_repeating_scope(body.id);
    intravisit::walk_expr(rcx, expr);
    rcx.set_repeating_scope(repeating_scope);
}

fn constrain_callee(rcx: &mut Rcx,
                    callee_id: ast::NodeId,
                    _call_expr: &hir::Expr,
                    _callee_expr: &hir::Expr) {
    let callee_ty = rcx.resolve_node_type(callee_id);
    match callee_ty.sty {
        ty::TyBareFn(..) => { }
        _ => {
            // this should not happen, but it does if the program is
            // erroneous
            //
            // tcx.sess.span_bug(
            //     callee_expr.span,
            //     format!("Calling non-function: {}", callee_ty));
        }
    }
}

fn constrain_call<'a, I: Iterator<Item=&'a hir::Expr>>(rcx: &mut Rcx,
                                                       call_expr: &hir::Expr,
                                                       receiver: Option<&hir::Expr>,
                                                       arg_exprs: I,
                                                       implicitly_ref_args: bool) {
    //! Invoked on every call site (i.e., normal calls, method calls,
    //! and overloaded operators). Constrains the regions which appear
    //! in the type of the function. Also constrains the regions that
    //! appear in the arguments appropriately.

    debug!("constrain_call(call_expr={:?}, \
            receiver={:?}, \
            implicitly_ref_args={})",
            call_expr,
            receiver,
            implicitly_ref_args);

    // `callee_region` is the scope representing the time in which the
    // call occurs.
    //
    // FIXME(#6268) to support nested method calls, should be callee_id
    let callee_scope = rcx.tcx().region_maps.node_extent(call_expr.id);
    let callee_region = ty::ReScope(callee_scope);

    debug!("callee_region={:?}", callee_region);

    for arg_expr in arg_exprs {
        debug!("Argument: {:?}", arg_expr);

        // ensure that any regions appearing in the argument type are
        // valid for at least the lifetime of the function:
        type_of_node_must_outlive(
            rcx, infer::CallArg(arg_expr.span),
            arg_expr.id, callee_region);

        // unfortunately, there are two means of taking implicit
        // references, and we need to propagate constraints as a
        // result. modes are going away and the "DerefArgs" code
        // should be ported to use adjustments
        if implicitly_ref_args {
            link_by_ref(rcx, arg_expr, callee_scope);
        }
    }

    // as loop above, but for receiver
    if let Some(r) = receiver {
        debug!("receiver: {:?}", r);
        type_of_node_must_outlive(
            rcx, infer::CallRcvr(r.span),
            r.id, callee_region);
        if implicitly_ref_args {
            link_by_ref(rcx, &*r, callee_scope);
        }
    }
}

/// Invoked on any auto-dereference that occurs. Checks that if this is a region pointer being
/// dereferenced, the lifetime of the pointer includes the deref expr.
fn constrain_autoderefs<'a, 'tcx>(rcx: &mut Rcx<'a, 'tcx>,
                                  deref_expr: &hir::Expr,
                                  derefs: usize,
                                  mut derefd_ty: Ty<'tcx>)
{
    debug!("constrain_autoderefs(deref_expr={:?}, derefs={}, derefd_ty={:?})",
           deref_expr,
           derefs,
           derefd_ty);

    let s_deref_expr = rcx.tcx().region_maps.node_extent(deref_expr.id);
    let r_deref_expr = ty::ReScope(s_deref_expr);
    for i in 0..derefs {
        let method_call = MethodCall::autoderef(deref_expr.id, i as u32);
        debug!("constrain_autoderefs: method_call={:?} (of {:?} total)", method_call, derefs);

        let method = rcx.fcx.inh.tables.borrow().method_map.get(&method_call).map(|m| m.clone());

        derefd_ty = match method {
            Some(method) => {
                debug!("constrain_autoderefs: #{} is overloaded, method={:?}",
                       i, method);

                let origin = infer::ParameterOrigin::OverloadedDeref;
                substs_wf_in_scope(rcx, origin, method.substs, deref_expr.span, r_deref_expr);

                // Treat overloaded autoderefs as if an AutoRef adjustment
                // was applied on the base type, as that is always the case.
                let fn_sig = method.ty.fn_sig();
                let fn_sig = // late-bound regions should have been instantiated
                    rcx.tcx().no_late_bound_regions(fn_sig).unwrap();
                let self_ty = fn_sig.inputs[0];
                let (m, r) = match self_ty.sty {
                    ty::TyRef(r, ref m) => (m.mutbl, r),
                    _ => {
                        rcx.tcx().sess.span_bug(
                            deref_expr.span,
                            &format!("bad overloaded deref type {:?}",
                                     method.ty))
                    }
                };

                debug!("constrain_autoderefs: receiver r={:?} m={:?}",
                       r, m);

                {
                    let mc = mc::MemCategorizationContext::new(rcx.fcx.infcx());
                    let self_cmt = ignore_err!(mc.cat_expr_autoderefd(deref_expr, i));
                    debug!("constrain_autoderefs: self_cmt={:?}",
                           self_cmt);
                    link_region(rcx, deref_expr.span, r,
                                ty::BorrowKind::from_mutbl(m), self_cmt);
                }

                // Specialized version of constrain_call.
                type_must_outlive(rcx, infer::CallRcvr(deref_expr.span),
                                  self_ty, r_deref_expr);
                match fn_sig.output {
                    ty::FnConverging(return_type) => {
                        type_must_outlive(rcx, infer::CallReturn(deref_expr.span),
                                          return_type, r_deref_expr);
                        return_type
                    }
                    ty::FnDiverging => unreachable!()
                }
            }
            None => derefd_ty
        };

        if let ty::TyRef(r_ptr, _) =  derefd_ty.sty {
            mk_subregion_due_to_dereference(rcx, deref_expr.span,
                                            r_deref_expr, *r_ptr);
        }

        match derefd_ty.builtin_deref(true, ty::NoPreference) {
            Some(mt) => derefd_ty = mt.ty,
            /* if this type can't be dereferenced, then there's already an error
               in the session saying so. Just bail out for now */
            None => break
        }
    }
}

pub fn mk_subregion_due_to_dereference(rcx: &mut Rcx,
                                       deref_span: Span,
                                       minimum_lifetime: ty::Region,
                                       maximum_lifetime: ty::Region) {
    rcx.fcx.mk_subr(infer::DerefPointer(deref_span),
                    minimum_lifetime, maximum_lifetime)
}

fn check_safety_of_rvalue_destructor_if_necessary<'a, 'tcx>(rcx: &mut Rcx<'a, 'tcx>,
                                                            cmt: mc::cmt<'tcx>,
                                                            span: Span) {
    match cmt.cat {
        Categorization::Rvalue(region) => {
            match region {
                ty::ReScope(rvalue_scope) => {
                    let typ = rcx.resolve_type(cmt.ty);
                    dropck::check_safety_of_destructor_if_necessary(rcx,
                                                                    typ,
                                                                    span,
                                                                    rvalue_scope);
                }
                ty::ReStatic => {}
                region => {
                    rcx.tcx()
                       .sess
                       .span_bug(span,
                                 &format!("unexpected rvalue region in rvalue \
                                           destructor safety checking: `{:?}`",
                                          region));
                }
            }
        }
        _ => {}
    }
}

/// Invoked on any index expression that occurs. Checks that if this is a slice being indexed, the
/// lifetime of the pointer includes the deref expr.
fn constrain_index<'a, 'tcx>(rcx: &mut Rcx<'a, 'tcx>,
                             index_expr: &hir::Expr,
                             indexed_ty: Ty<'tcx>)
{
    debug!("constrain_index(index_expr=?, indexed_ty={}",
           rcx.fcx.infcx().ty_to_string(indexed_ty));

    let r_index_expr = ty::ReScope(rcx.tcx().region_maps.node_extent(index_expr.id));
    if let ty::TyRef(r_ptr, mt) = indexed_ty.sty {
        match mt.ty.sty {
            ty::TySlice(_) | ty::TyStr => {
                rcx.fcx.mk_subr(infer::IndexSlice(index_expr.span),
                                r_index_expr, *r_ptr);
            }
            _ => {}
        }
    }
}

/// Guarantees that any lifetimes which appear in the type of the node `id` (after applying
/// adjustments) are valid for at least `minimum_lifetime`
fn type_of_node_must_outlive<'a, 'tcx>(
    rcx: &mut Rcx<'a, 'tcx>,
    origin: infer::SubregionOrigin<'tcx>,
    id: ast::NodeId,
    minimum_lifetime: ty::Region)
{
    let tcx = rcx.fcx.tcx();

    // Try to resolve the type.  If we encounter an error, then typeck
    // is going to fail anyway, so just stop here and let typeck
    // report errors later on in the writeback phase.
    let ty0 = rcx.resolve_node_type(id);
    let ty = ty0.adjust(tcx, origin.span(), id,
                        rcx.fcx.inh.tables.borrow().adjustments.get(&id),
                        |method_call| rcx.resolve_method_type(method_call));
    debug!("constrain_regions_in_type_of_node(\
            ty={}, ty0={}, id={}, minimum_lifetime={:?})",
            ty,  ty0,
           id, minimum_lifetime);
    type_must_outlive(rcx, origin, ty, minimum_lifetime);
}

/// Computes the guarantor for an expression `&base` and then ensures that the lifetime of the
/// resulting pointer is linked to the lifetime of its guarantor (if any).
fn link_addr_of(rcx: &mut Rcx, expr: &hir::Expr,
                mutability: hir::Mutability, base: &hir::Expr) {
    debug!("link_addr_of(expr={:?}, base={:?})", expr, base);

    let cmt = {
        let mc = mc::MemCategorizationContext::new(rcx.fcx.infcx());
        ignore_err!(mc.cat_expr(base))
    };

    debug!("link_addr_of: cmt={:?}", cmt);

    link_region_from_node_type(rcx, expr.span, expr.id, mutability, cmt);
}

/// Computes the guarantors for any ref bindings in a `let` and
/// then ensures that the lifetime of the resulting pointer is
/// linked to the lifetime of the initialization expression.
fn link_local(rcx: &Rcx, local: &hir::Local) {
    debug!("regionck::for_local()");
    let init_expr = match local.init {
        None => { return; }
        Some(ref expr) => &**expr,
    };
    let mc = mc::MemCategorizationContext::new(rcx.fcx.infcx());
    let discr_cmt = ignore_err!(mc.cat_expr(init_expr));
    link_pattern(rcx, mc, discr_cmt, &*local.pat);
}

/// Computes the guarantors for any ref bindings in a match and
/// then ensures that the lifetime of the resulting pointer is
/// linked to the lifetime of its guarantor (if any).
fn link_match(rcx: &Rcx, discr: &hir::Expr, arms: &[hir::Arm]) {
    debug!("regionck::for_match()");
    let mc = mc::MemCategorizationContext::new(rcx.fcx.infcx());
    let discr_cmt = ignore_err!(mc.cat_expr(discr));
    debug!("discr_cmt={:?}", discr_cmt);
    for arm in arms {
        for root_pat in &arm.pats {
            link_pattern(rcx, mc, discr_cmt.clone(), &**root_pat);
        }
    }
}

/// Computes the guarantors for any ref bindings in a match and
/// then ensures that the lifetime of the resulting pointer is
/// linked to the lifetime of its guarantor (if any).
fn link_fn_args(rcx: &Rcx, body_scope: CodeExtent, args: &[hir::Arg]) {
    debug!("regionck::link_fn_args(body_scope={:?})", body_scope);
    let mc = mc::MemCategorizationContext::new(rcx.fcx.infcx());
    for arg in args {
        let arg_ty = rcx.fcx.node_ty(arg.id);
        let re_scope = ty::ReScope(body_scope);
        let arg_cmt = mc.cat_rvalue(arg.id, arg.ty.span, re_scope, arg_ty);
        debug!("arg_ty={:?} arg_cmt={:?} arg={:?}",
               arg_ty,
               arg_cmt,
               arg);
        link_pattern(rcx, mc, arg_cmt, &*arg.pat);
    }
}

/// Link lifetimes of any ref bindings in `root_pat` to the pointers found in the discriminant, if
/// needed.
fn link_pattern<'t, 'a, 'tcx>(rcx: &Rcx<'a, 'tcx>,
                          mc: mc::MemCategorizationContext<'t, 'a, 'tcx>,
                          discr_cmt: mc::cmt<'tcx>,
                          root_pat: &hir::Pat) {
    debug!("link_pattern(discr_cmt={:?}, root_pat={:?})",
           discr_cmt,
           root_pat);
    let _ = mc.cat_pattern(discr_cmt, root_pat, |mc, sub_cmt, sub_pat| {
            match sub_pat.node {
                // `ref x` pattern
                hir::PatIdent(hir::BindByRef(mutbl), _, _) => {
                    link_region_from_node_type(
                        rcx, sub_pat.span, sub_pat.id,
                        mutbl, sub_cmt);
                }

                // `[_, ..slice, _]` pattern
                hir::PatVec(_, Some(ref slice_pat), _) => {
                    match mc.cat_slice_pattern(sub_cmt, &**slice_pat) {
                        Ok((slice_cmt, slice_mutbl, slice_r)) => {
                            link_region(rcx, sub_pat.span, &slice_r,
                                        ty::BorrowKind::from_mutbl(slice_mutbl),
                                        slice_cmt);
                        }
                        Err(()) => {}
                    }
                }
                _ => {}
            }
        });
}

/// Link lifetime of borrowed pointer resulting from autoref to lifetimes in the value being
/// autoref'd.
fn link_autoref(rcx: &Rcx,
                expr: &hir::Expr,
                autoderefs: usize,
                autoref: &adjustment::AutoRef)
{
    debug!("link_autoref(autoref={:?})", autoref);
    let mc = mc::MemCategorizationContext::new(rcx.fcx.infcx());
    let expr_cmt = ignore_err!(mc.cat_expr_autoderefd(expr, autoderefs));
    debug!("expr_cmt={:?}", expr_cmt);

    match *autoref {
        adjustment::AutoPtr(r, m) => {
            link_region(rcx, expr.span, r,
                ty::BorrowKind::from_mutbl(m), expr_cmt);
        }

        adjustment::AutoUnsafe(m) => {
            let r = ty::ReScope(rcx.tcx().region_maps.node_extent(expr.id));
            link_region(rcx, expr.span, &r, ty::BorrowKind::from_mutbl(m), expr_cmt);
        }
    }
}

/// Computes the guarantor for cases where the `expr` is being passed by implicit reference and
/// must outlive `callee_scope`.
fn link_by_ref(rcx: &Rcx,
               expr: &hir::Expr,
               callee_scope: CodeExtent) {
    debug!("link_by_ref(expr={:?}, callee_scope={:?})",
           expr, callee_scope);
    let mc = mc::MemCategorizationContext::new(rcx.fcx.infcx());
    let expr_cmt = ignore_err!(mc.cat_expr(expr));
    let borrow_region = ty::ReScope(callee_scope);
    link_region(rcx, expr.span, &borrow_region, ty::ImmBorrow, expr_cmt);
}

/// Like `link_region()`, except that the region is extracted from the type of `id`, which must be
/// some reference (`&T`, `&str`, etc).
fn link_region_from_node_type<'a, 'tcx>(rcx: &Rcx<'a, 'tcx>,
                                        span: Span,
                                        id: ast::NodeId,
                                        mutbl: hir::Mutability,
                                        cmt_borrowed: mc::cmt<'tcx>) {
    debug!("link_region_from_node_type(id={:?}, mutbl={:?}, cmt_borrowed={:?})",
           id, mutbl, cmt_borrowed);

    let rptr_ty = rcx.resolve_node_type(id);
    if let ty::TyRef(&r, _) = rptr_ty.sty {
        debug!("rptr_ty={}",  rptr_ty);
        link_region(rcx, span, &r, ty::BorrowKind::from_mutbl(mutbl),
                    cmt_borrowed);
    }
}

/// Informs the inference engine that `borrow_cmt` is being borrowed with kind `borrow_kind` and
/// lifetime `borrow_region`. In order to ensure borrowck is satisfied, this may create constraints
/// between regions, as explained in `link_reborrowed_region()`.
fn link_region<'a, 'tcx>(rcx: &Rcx<'a, 'tcx>,
                         span: Span,
                         borrow_region: &ty::Region,
                         borrow_kind: ty::BorrowKind,
                         borrow_cmt: mc::cmt<'tcx>) {
    let mut borrow_cmt = borrow_cmt;
    let mut borrow_kind = borrow_kind;

    let origin = infer::DataBorrowed(borrow_cmt.ty, span);
    type_must_outlive(rcx, origin, borrow_cmt.ty, *borrow_region);

    loop {
        debug!("link_region(borrow_region={:?}, borrow_kind={:?}, borrow_cmt={:?})",
               borrow_region,
               borrow_kind,
               borrow_cmt);
        match borrow_cmt.cat.clone() {
            Categorization::Deref(ref_cmt, _,
                                  mc::Implicit(ref_kind, ref_region)) |
            Categorization::Deref(ref_cmt, _,
                                  mc::BorrowedPtr(ref_kind, ref_region)) => {
                match link_reborrowed_region(rcx, span,
                                             borrow_region, borrow_kind,
                                             ref_cmt, ref_region, ref_kind,
                                             borrow_cmt.note) {
                    Some((c, k)) => {
                        borrow_cmt = c;
                        borrow_kind = k;
                    }
                    None => {
                        return;
                    }
                }
            }

            Categorization::Downcast(cmt_base, _) |
            Categorization::Deref(cmt_base, _, mc::Unique) |
            Categorization::Interior(cmt_base, _) => {
                // Borrowing interior or owned data requires the base
                // to be valid and borrowable in the same fashion.
                borrow_cmt = cmt_base;
                borrow_kind = borrow_kind;
            }

            Categorization::Deref(_, _, mc::UnsafePtr(..)) |
            Categorization::StaticItem |
            Categorization::Upvar(..) |
            Categorization::Local(..) |
            Categorization::Rvalue(..) => {
                // These are all "base cases" with independent lifetimes
                // that are not subject to inference
                return;
            }
        }
    }
}

/// This is the most complicated case: the path being borrowed is
/// itself the referent of a borrowed pointer. Let me give an
/// example fragment of code to make clear(er) the situation:
///
///    let r: &'a mut T = ...;  // the original reference "r" has lifetime 'a
///    ...
///    &'z *r                   // the reborrow has lifetime 'z
///
/// Now, in this case, our primary job is to add the inference
/// constraint that `'z <= 'a`. Given this setup, let's clarify the
/// parameters in (roughly) terms of the example:
///
///     A borrow of: `& 'z bk * r` where `r` has type `& 'a bk T`
///     borrow_region   ^~                 ref_region    ^~
///     borrow_kind        ^~               ref_kind        ^~
///     ref_cmt                 ^
///
/// Here `bk` stands for some borrow-kind (e.g., `mut`, `uniq`, etc).
///
/// Unfortunately, there are some complications beyond the simple
/// scenario I just painted:
///
/// 1. The reference `r` might in fact be a "by-ref" upvar. In that
///    case, we have two jobs. First, we are inferring whether this reference
///    should be an `&T`, `&mut T`, or `&uniq T` reference, and we must
///    adjust that based on this borrow (e.g., if this is an `&mut` borrow,
///    then `r` must be an `&mut` reference). Second, whenever we link
///    two regions (here, `'z <= 'a`), we supply a *cause*, and in this
///    case we adjust the cause to indicate that the reference being
///    "reborrowed" is itself an upvar. This provides a nicer error message
///    should something go wrong.
///
/// 2. There may in fact be more levels of reborrowing. In the
///    example, I said the borrow was like `&'z *r`, but it might
///    in fact be a borrow like `&'z **q` where `q` has type `&'a
///    &'b mut T`. In that case, we want to ensure that `'z <= 'a`
///    and `'z <= 'b`. This is explained more below.
///
/// The return value of this function indicates whether we need to
/// recurse and process `ref_cmt` (see case 2 above).
fn link_reborrowed_region<'a, 'tcx>(rcx: &Rcx<'a, 'tcx>,
                                    span: Span,
                                    borrow_region: &ty::Region,
                                    borrow_kind: ty::BorrowKind,
                                    ref_cmt: mc::cmt<'tcx>,
                                    ref_region: ty::Region,
                                    mut ref_kind: ty::BorrowKind,
                                    note: mc::Note)
                                    -> Option<(mc::cmt<'tcx>, ty::BorrowKind)>
{
    // Possible upvar ID we may need later to create an entry in the
    // maybe link map.

    // Detect by-ref upvar `x`:
    let cause = match note {
        mc::NoteUpvarRef(ref upvar_id) => {
            let upvar_capture_map = &rcx.fcx.inh.tables.borrow_mut().upvar_capture_map;
            match upvar_capture_map.get(upvar_id) {
                Some(&ty::UpvarCapture::ByRef(ref upvar_borrow)) => {
                    // The mutability of the upvar may have been modified
                    // by the above adjustment, so update our local variable.
                    ref_kind = upvar_borrow.kind;

                    infer::ReborrowUpvar(span, *upvar_id)
                }
                _ => {
                    rcx.tcx().sess.span_bug(
                        span,
                        &format!("Illegal upvar id: {:?}",
                                upvar_id));
                }
            }
        }
        mc::NoteClosureEnv(ref upvar_id) => {
            // We don't have any mutability changes to propagate, but
            // we do want to note that an upvar reborrow caused this
            // link
            infer::ReborrowUpvar(span, *upvar_id)
        }
        _ => {
            infer::Reborrow(span)
        }
    };

    debug!("link_reborrowed_region: {:?} <= {:?}",
           borrow_region,
           ref_region);
    rcx.fcx.mk_subr(cause, *borrow_region, ref_region);

    // If we end up needing to recurse and establish a region link
    // with `ref_cmt`, calculate what borrow kind we will end up
    // needing. This will be used below.
    //
    // One interesting twist is that we can weaken the borrow kind
    // when we recurse: to reborrow an `&mut` referent as mutable,
    // borrowck requires a unique path to the `&mut` reference but not
    // necessarily a *mutable* path.
    let new_borrow_kind = match borrow_kind {
        ty::ImmBorrow =>
            ty::ImmBorrow,
        ty::MutBorrow | ty::UniqueImmBorrow =>
            ty::UniqueImmBorrow
    };

    // Decide whether we need to recurse and link any regions within
    // the `ref_cmt`. This is concerned for the case where the value
    // being reborrowed is in fact a borrowed pointer found within
    // another borrowed pointer. For example:
    //
    //    let p: &'b &'a mut T = ...;
    //    ...
    //    &'z **p
    //
    // What makes this case particularly tricky is that, if the data
    // being borrowed is a `&mut` or `&uniq` borrow, borrowck requires
    // not only that `'z <= 'a`, (as before) but also `'z <= 'b`
    // (otherwise the user might mutate through the `&mut T` reference
    // after `'b` expires and invalidate the borrow we are looking at
    // now).
    //
    // So let's re-examine our parameters in light of this more
    // complicated (possible) scenario:
    //
    //     A borrow of: `& 'z bk * * p` where `p` has type `&'b bk & 'a bk T`
    //     borrow_region   ^~                 ref_region             ^~
    //     borrow_kind        ^~               ref_kind                 ^~
    //     ref_cmt                 ^~~
    //
    // (Note that since we have not examined `ref_cmt.cat`, we don't
    // know whether this scenario has occurred; but I wanted to show
    // how all the types get adjusted.)
    match ref_kind {
        ty::ImmBorrow => {
            // The reference being reborrowed is a sharable ref of
            // type `&'a T`. In this case, it doesn't matter where we
            // *found* the `&T` pointer, the memory it references will
            // be valid and immutable for `'a`. So we can stop here.
            //
            // (Note that the `borrow_kind` must also be ImmBorrow or
            // else the user is borrowed imm memory as mut memory,
            // which means they'll get an error downstream in borrowck
            // anyhow.)
            return None;
        }

        ty::MutBorrow | ty::UniqueImmBorrow => {
            // The reference being reborrowed is either an `&mut T` or
            // `&uniq T`. This is the case where recursion is needed.
            return Some((ref_cmt, new_borrow_kind));
        }
    }
}

/// Checks that the values provided for type/region arguments in a given
/// expression are well-formed and in-scope.
pub fn substs_wf_in_scope<'a,'tcx>(rcx: &mut Rcx<'a,'tcx>,
                                   origin: infer::ParameterOrigin,
                                   substs: &Substs<'tcx>,
                                   expr_span: Span,
                                   expr_region: ty::Region) {
    debug!("substs_wf_in_scope(substs={:?}, \
            expr_region={:?}, \
            origin={:?}, \
            expr_span={:?})",
           substs, expr_region, origin, expr_span);

    let origin = infer::ParameterInScope(origin, expr_span);

    for &region in substs.regions() {
        rcx.fcx.mk_subr(origin.clone(), expr_region, region);
    }

    for &ty in &substs.types {
        let ty = rcx.resolve_type(ty);
        type_must_outlive(rcx, origin.clone(), ty, expr_region);
    }
}

/// Ensures that type is well-formed in `region`, which implies (among
/// other things) that all borrowed data reachable via `ty` outlives
/// `region`.
pub fn type_must_outlive<'a, 'tcx>(rcx: &Rcx<'a, 'tcx>,
                                   origin: infer::SubregionOrigin<'tcx>,
                                   ty: Ty<'tcx>,
                                   region: ty::Region)
{
    let ty = rcx.resolve_type(ty);

    debug!("type_must_outlive(ty={:?}, region={:?}, origin={:?})",
           ty,
           region,
           origin);

    assert!(!ty.has_escaping_regions());

    let components = ty::outlives::components(rcx.infcx(), ty);
    components_must_outlive(rcx, origin, components, region);
}

fn components_must_outlive<'a, 'tcx>(rcx: &Rcx<'a, 'tcx>,
                                     origin: infer::SubregionOrigin<'tcx>,
                                     components: Vec<ty::outlives::Component<'tcx>>,
                                     region: ty::Region)
{
    for component in components {
        let origin = origin.clone();
        match component {
            ty::outlives::Component::Region(region1) => {
                rcx.fcx.mk_subr(origin, region, region1);
            }
            ty::outlives::Component::Param(param_ty) => {
                param_ty_must_outlive(rcx, origin, region, param_ty);
            }
            ty::outlives::Component::Projection(projection_ty) => {
                projection_must_outlive(rcx, origin, region, projection_ty);
            }
            ty::outlives::Component::EscapingProjection(subcomponents) => {
                components_must_outlive(rcx, origin, subcomponents, region);
            }
            ty::outlives::Component::UnresolvedInferenceVariable(v) => {
                // ignore this, we presume it will yield an error
                // later, since if a type variable is not resolved by
                // this point it never will be
                rcx.tcx().sess.delay_span_bug(
                    origin.span(),
                    &format!("unresolved inference variable in outlives: {:?}", v));
            }
        }
    }
}

fn param_ty_must_outlive<'a, 'tcx>(rcx: &Rcx<'a, 'tcx>,
                                   origin: infer::SubregionOrigin<'tcx>,
                                   region: ty::Region,
                                   param_ty: ty::ParamTy) {
    debug!("param_ty_must_outlive(region={:?}, param_ty={:?}, origin={:?})",
           region, param_ty, origin);

    let verify_bound = param_bound(rcx, param_ty);
    let generic = GenericKind::Param(param_ty);
    rcx.fcx.infcx().verify_generic_bound(origin, generic, region, verify_bound);
}

fn projection_must_outlive<'a, 'tcx>(rcx: &Rcx<'a, 'tcx>,
                                     origin: infer::SubregionOrigin<'tcx>,
                                     region: ty::Region,
                                     projection_ty: ty::ProjectionTy<'tcx>)
{
    debug!("projection_must_outlive(region={:?}, projection_ty={:?}, origin={:?})",
           region, projection_ty, origin);

    // This case is thorny for inference. The fundamental problem is
    // that there are many cases where we have choice, and inference
    // doesn't like choice (the current region inference in
    // particular). :) First off, we have to choose between using the
    // OutlivesProjectionEnv, OutlivesProjectionTraitDef, and
    // OutlivesProjectionComponent rules, any one of which is
    // sufficient.  If there are no inference variables involved, it's
    // not hard to pick the right rule, but if there are, we're in a
    // bit of a catch 22: if we picked which rule we were going to
    // use, we could add constraints to the region inference graph
    // that make it apply, but if we don't add those constraints, the
    // rule might not apply (but another rule might). For now, we err
    // on the side of adding too few edges into the graph.

    // Compute the bounds we can derive from the environment or trait
    // definition.  We know that the projection outlives all the
    // regions in this list.
    let env_bounds = projection_declared_bounds(rcx, origin.span(), projection_ty);

    debug!("projection_must_outlive: env_bounds={:?}",
           env_bounds);

    // If we know that the projection outlives 'static, then we're
    // done here.
    if env_bounds.contains(&ty::ReStatic) {
        debug!("projection_must_outlive: 'static as declared bound");
        return;
    }

    // If declared bounds list is empty, the only applicable rule is
    // OutlivesProjectionComponent. If there are inference variables,
    // then, we can break down the outlives into more primitive
    // components without adding unnecessary edges.
    //
    // If there are *no* inference variables, however, we COULD do
    // this, but we choose not to, because the error messages are less
    // good. For example, a requirement like `T::Item: 'r` would be
    // translated to a requirement that `T: 'r`; when this is reported
    // to the user, it will thus say "T: 'r must hold so that T::Item:
    // 'r holds". But that makes it sound like the only way to fix
    // the problem is to add `T: 'r`, which isn't true. So, if there are no
    // inference variables, we use a verify constraint instead of adding
    // edges, which winds up enforcing the same condition.
    let needs_infer = {
        projection_ty.trait_ref.substs.types.iter().any(|t| t.needs_infer()) ||
            projection_ty.trait_ref.substs.regions().iter().any(|r| r.needs_infer())
    };
    if env_bounds.is_empty() && needs_infer {
        debug!("projection_must_outlive: no declared bounds");

        for &component_ty in &projection_ty.trait_ref.substs.types {
            type_must_outlive(rcx, origin.clone(), component_ty, region);
        }

        for &r in projection_ty.trait_ref.substs.regions() {
            rcx.fcx.mk_subr(origin.clone(), region, r);
        }

        return;
    }

    // If we find that there is a unique declared bound `'b`, and this bound
    // appears in the trait reference, then the best action is to require that `'b:'r`,
    // so do that. This is best no matter what rule we use:
    //
    // - OutlivesProjectionEnv or OutlivesProjectionTraitDef: these would translate to
    // the requirement that `'b:'r`
    // - OutlivesProjectionComponent: this would require `'b:'r` in addition to other conditions
    if !env_bounds.is_empty() && env_bounds[1..].iter().all(|b| *b == env_bounds[0]) {
        let unique_bound = env_bounds[0];
        debug!("projection_must_outlive: unique declared bound = {:?}", unique_bound);
        if projection_ty.trait_ref.substs.regions()
                                         .iter()
                                         .any(|r| env_bounds.contains(r))
        {
            debug!("projection_must_outlive: unique declared bound appears in trait ref");
            rcx.fcx.mk_subr(origin.clone(), region, unique_bound);
            return;
        }
    }

    // Fallback to verifying after the fact that there exists a
    // declared bound, or that all the components appearing in the
    // projection outlive; in some cases, this may add insufficient
    // edges into the inference graph, leading to inference failures
    // even though a satisfactory solution exists.
    let verify_bound = projection_bound(rcx, origin.span(), env_bounds, projection_ty);
    let generic = GenericKind::Projection(projection_ty);
    rcx.fcx.infcx().verify_generic_bound(origin, generic.clone(), region, verify_bound);
}

fn type_bound<'a, 'tcx>(rcx: &Rcx<'a, 'tcx>, span: Span, ty: Ty<'tcx>) -> VerifyBound {
    match ty.sty {
        ty::TyParam(p) => {
            param_bound(rcx, p)
        }
        ty::TyProjection(data) => {
            let declared_bounds = projection_declared_bounds(rcx, span, data);
            projection_bound(rcx, span, declared_bounds, data)
        }
        _ => {
            recursive_type_bound(rcx, span, ty)
        }
    }
}

fn param_bound<'a, 'tcx>(rcx: &Rcx<'a, 'tcx>, param_ty: ty::ParamTy) -> VerifyBound {
    let param_env = &rcx.infcx().parameter_environment;

    debug!("param_bound(param_ty={:?})",
           param_ty);

    let mut param_bounds = declared_generic_bounds_from_env(rcx, GenericKind::Param(param_ty));

    // Add in the default bound of fn body that applies to all in
    // scope type parameters:
    param_bounds.push(param_env.implicit_region_bound);

    VerifyBound::AnyRegion(param_bounds)
}

fn projection_declared_bounds<'a, 'tcx>(rcx: &Rcx<'a,'tcx>,
                                        span: Span,
                                        projection_ty: ty::ProjectionTy<'tcx>)
                                        -> Vec<ty::Region>
{
    // First assemble bounds from where clauses and traits.

    let mut declared_bounds =
        declared_generic_bounds_from_env(rcx, GenericKind::Projection(projection_ty));

    declared_bounds.extend_from_slice(
        &declared_projection_bounds_from_trait(rcx, span, projection_ty));

    declared_bounds
}

fn projection_bound<'a, 'tcx>(rcx: &Rcx<'a, 'tcx>,
                              span: Span,
                              declared_bounds: Vec<ty::Region>,
                              projection_ty: ty::ProjectionTy<'tcx>)
                              -> VerifyBound {
    debug!("projection_bound(declared_bounds={:?}, projection_ty={:?})",
           declared_bounds, projection_ty);

    // see the extensive comment in projection_must_outlive

    let ty = rcx.tcx().mk_projection(projection_ty.trait_ref, projection_ty.item_name);
    let recursive_bound = recursive_type_bound(rcx, span, ty);

    VerifyBound::AnyRegion(declared_bounds).or(recursive_bound)
}

fn recursive_type_bound<'a, 'tcx>(rcx: &Rcx<'a, 'tcx>,
                                  span: Span,
                                  ty: Ty<'tcx>)
                                  -> VerifyBound {
    let mut bounds = vec![];

    for subty in ty.walk_shallow() {
        bounds.push(type_bound(rcx, span, subty));
    }

    let mut regions = ty.regions();
    regions.retain(|r| !r.is_bound()); // ignore late-bound regions
    bounds.push(VerifyBound::AllRegions(regions));

    // remove bounds that must hold, since they are not interesting
    bounds.retain(|b| !b.must_hold());

    if bounds.len() == 1 {
        bounds.pop().unwrap()
    } else {
        VerifyBound::AllBounds(bounds)
    }
}

fn declared_generic_bounds_from_env<'a, 'tcx>(rcx: &Rcx<'a, 'tcx>,
                                              generic: GenericKind<'tcx>)
                                              -> Vec<ty::Region>
{
    let param_env = &rcx.infcx().parameter_environment;

    // To start, collect bounds from user:
    let mut param_bounds = rcx.tcx().required_region_bounds(generic.to_ty(rcx.tcx()),
                                                            param_env.caller_bounds.clone());

    // Next, collect regions we scraped from the well-formedness
    // constraints in the fn signature. To do that, we walk the list
    // of known relations from the fn ctxt.
    //
    // This is crucial because otherwise code like this fails:
    //
    //     fn foo<'a, A>(x: &'a A) { x.bar() }
    //
    // The problem is that the type of `x` is `&'a A`. To be
    // well-formed, then, A must be lower-generic by `'a`, but we
    // don't know that this holds from first principles.
    for &(r, p) in &rcx.region_bound_pairs {
        debug!("generic={:?} p={:?}",
               generic,
               p);
        if generic == p {
            param_bounds.push(r);
        }
    }

    param_bounds
}

fn declared_projection_bounds_from_trait<'a,'tcx>(rcx: &Rcx<'a, 'tcx>,
                                                  span: Span,
                                                  projection_ty: ty::ProjectionTy<'tcx>)
                                                  -> Vec<ty::Region>
{
    let fcx = rcx.fcx;
    let tcx = fcx.tcx();
    let infcx = fcx.infcx();

    debug!("projection_bounds(projection_ty={:?})",
           projection_ty);

    let ty = tcx.mk_projection(projection_ty.trait_ref.clone(), projection_ty.item_name);

    // Say we have a projection `<T as SomeTrait<'a>>::SomeType`. We are interested
    // in looking for a trait definition like:
    //
    // ```
    // trait SomeTrait<'a> {
    //     type SomeType : 'a;
    // }
    // ```
    //
    // we can thus deduce that `<T as SomeTrait<'a>>::SomeType : 'a`.
    let trait_predicates = tcx.lookup_predicates(projection_ty.trait_ref.def_id);
    let predicates = trait_predicates.predicates.as_slice().to_vec();
    traits::elaborate_predicates(tcx, predicates)
        .filter_map(|predicate| {
            // we're only interesting in `T : 'a` style predicates:
            let outlives = match predicate {
                ty::Predicate::TypeOutlives(data) => data,
                _ => { return None; }
            };

            debug!("projection_bounds: outlives={:?} (1)",
                   outlives);

            // apply the substitutions (and normalize any projected types)
            let outlives = fcx.instantiate_type_scheme(span,
                                                       projection_ty.trait_ref.substs,
                                                       &outlives);

            debug!("projection_bounds: outlives={:?} (2)",
                   outlives);

            let region_result = infcx.commit_if_ok(|_| {
                let (outlives, _) =
                    infcx.replace_late_bound_regions_with_fresh_var(
                        span,
                        infer::AssocTypeProjection(projection_ty.item_name),
                        &outlives);

                debug!("projection_bounds: outlives={:?} (3)",
                       outlives);

                // check whether this predicate applies to our current projection
                match infer::mk_eqty(infcx, false, TypeOrigin::Misc(span), ty, outlives.0) {
                    Ok(()) => { Ok(outlives.1) }
                    Err(_) => { Err(()) }
                }
            });

            debug!("projection_bounds: region_result={:?}",
                   region_result);

            region_result.ok()
        })
        .collect()
}
