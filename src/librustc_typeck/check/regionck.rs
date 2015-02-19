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
//!     struct Foo { i: int }
//!     struct Bar { foo: Foo  }
//!     fn get_i(x: &'a Bar) -> &'a int {
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
//! interior offsets or owned pointers.  We say that the guarantor
//! of such data it the region of the borrowed pointer that was
//! traversed.  This is essentially the same as the ownership
//! relation, except that a borrowed pointer never owns its
//! contents.

use astconv::AstConv;
use check::dropck;
use check::FnCtxt;
use check::implicator;
use check::vtable;
use middle::def;
use middle::mem_categorization as mc;
use middle::region::CodeExtent;
use middle::traits;
use middle::ty::{ReScope};
use middle::ty::{self, Ty, MethodCall};
use middle::infer::{self, GenericKind};
use middle::pat_util;
use util::ppaux::{ty_to_string, Repr};

use std::mem;
use syntax::{ast, ast_util};
use syntax::codemap::Span;
use syntax::visit;
use syntax::visit::Visitor;

use self::SubjectNode::Subject;

// a variation on try that just returns unit
macro_rules! ignore_err {
    ($e:expr) => (match $e { Ok(e) => e, Err(_) => return () })
}

///////////////////////////////////////////////////////////////////////////
// PUBLIC ENTRY POINTS

pub fn regionck_expr(fcx: &FnCtxt, e: &ast::Expr) {
    let mut rcx = Rcx::new(fcx, RepeatingScope(e.id), e.id, Subject(e.id));
    if fcx.err_count_since_creation() == 0 {
        // regionck assumes typeck succeeded
        rcx.visit_expr(e);
        rcx.visit_region_obligations(e.id);
    }
    rcx.resolve_regions_and_report_errors();
}

pub fn regionck_item(fcx: &FnCtxt, item: &ast::Item) {
    let mut rcx = Rcx::new(fcx, RepeatingScope(item.id), item.id, Subject(item.id));
    rcx.visit_region_obligations(item.id);
    rcx.resolve_regions_and_report_errors();
}

pub fn regionck_fn(fcx: &FnCtxt,
                   fn_id: ast::NodeId,
                   fn_span: Span,
                   decl: &ast::FnDecl,
                   blk: &ast::Block) {
    debug!("regionck_fn(id={})", fn_id);
    let mut rcx = Rcx::new(fcx, RepeatingScope(blk.id), blk.id, Subject(fn_id));
    if fcx.err_count_since_creation() == 0 {
        // regionck assumes typeck succeeded
        rcx.visit_fn_body(fn_id, decl, blk, fn_span);
    }

    rcx.resolve_regions_and_report_errors();
}

/// Checks that the types in `component_tys` are well-formed. This will add constraints into the
/// region graph. Does *not* run `resolve_regions_and_report_errors` and so forth.
pub fn regionck_ensure_component_tys_wf<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                                  span: Span,
                                                  component_tys: &[Ty<'tcx>]) {
    let mut rcx = Rcx::new(fcx, RepeatingScope(0), 0, SubjectNode::None);
    for &component_ty in component_tys {
        // Check that each type outlives the empty region. Since the
        // empty region is a subregion of all others, this can't fail
        // unless the type does not meet the well-formedness
        // requirements.
        type_must_outlive(&mut rcx, infer::RelateParamBound(span, component_ty),
                          component_ty, ty::ReEmpty);
    }
}

///////////////////////////////////////////////////////////////////////////
// INTERNALS

pub struct Rcx<'a, 'tcx: 'a> {
    fcx: &'a FnCtxt<'a, 'tcx>,

    region_bound_pairs: Vec<(ty::Region, GenericKind<'tcx>)>,

    // id of innermost fn body id
    body_id: ast::NodeId,

    // id of innermost fn or loop
    repeating_scope: ast::NodeId,

    // id of AST node being analyzed (the subject of the analysis).
    subject: SubjectNode,

}

/// Returns the validity region of `def` -- that is, how long is `def` valid?
fn region_of_def(fcx: &FnCtxt, def: def::Def) -> ty::Region {
    let tcx = fcx.tcx();
    match def {
        def::DefLocal(node_id) | def::DefUpvar(node_id, _) => {
            tcx.region_maps.var_region(node_id)
        }
        _ => {
            tcx.sess.bug(&format!("unexpected def in region_of_def: {:?}",
                                 def)[])
        }
    }
}

struct RepeatingScope(ast::NodeId);
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
              subject: subject,
              region_bound_pairs: Vec::new()
        }
    }

    pub fn tcx(&self) -> &'a ty::ctxt<'tcx> {
        self.fcx.ccx.tcx
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
    /// fn borrow(x: &int) -> &int {x}
    /// fn foo(x: @int) -> int {  // block: B
    ///     let b = borrow(x);    // region: <R0>
    ///     *b
    /// }
    /// ```
    ///
    /// Here, the region of `b` will be `<R0>`.  `<R0>` is constrainted to be some subregion of the
    /// block B and some superregion of the call.  If we forced it now, we'd choose the smaller
    /// region (the call).  But that would make the *b illegal.  Since we don't resolve, the type
    /// of b will be `&<R0>.int` and then `*b` will require that `<R0>` be bigger than the let and
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
        let method_ty = self.fcx.inh.method_map.borrow()
                            .get(&method_call).map(|method| method.ty);
        method_ty.map(|method_ty| self.resolve_type(method_ty))
    }

    /// Try to resolve the type for the given node.
    pub fn resolve_expr_type_adjusted(&mut self, expr: &ast::Expr) -> Ty<'tcx> {
        let ty_unadjusted = self.resolve_node_type(expr.id);
        if ty::type_is_error(ty_unadjusted) {
            ty_unadjusted
        } else {
            let tcx = self.fcx.tcx();
            ty::adjust_ty(tcx, expr.span, expr.id, ty_unadjusted,
                          self.fcx.inh.adjustments.borrow().get(&expr.id),
                          |method_call| self.resolve_method_type(method_call))
        }
    }

    fn visit_fn_body(&mut self,
                     id: ast::NodeId,
                     fn_decl: &ast::FnDecl,
                     body: &ast::Block,
                     span: Span)
    {
        // When we enter a function, we can derive
        debug!("visit_fn_body(id={})", id);

        let fn_sig_map = self.fcx.inh.fn_sig_map.borrow();
        let fn_sig = match fn_sig_map.get(&id) {
            Some(f) => f,
            None => {
                self.tcx().sess.bug(
                    &format!("No fn-sig entry for id={}", id)[]);
            }
        };

        let len = self.region_bound_pairs.len();
        let old_body_id = self.set_body_id(body.id);
        self.relate_free_regions(&fn_sig[..], body.id, span);
        link_fn_args(self, CodeExtent::from_node_id(body.id), &fn_decl.inputs[..]);
        self.visit_block(body);
        self.visit_region_obligations(body.id);
        self.region_bound_pairs.truncate(len);
        self.set_body_id(old_body_id);
    }

    fn visit_region_obligations(&mut self, node_id: ast::NodeId)
    {
        debug!("visit_region_obligations: node_id={}", node_id);

        // region checking can introduce new pending obligations
        // which, when processed, might generate new region
        // obligations. So make sure we process those.
        vtable::select_all_fcx_obligations_or_error(self.fcx);

        // Make a copy of the region obligations vec because we'll need
        // to be able to borrow the fulfillment-cx below when projecting.
        let region_obligations =
            self.fcx.inh.fulfillment_cx.borrow()
                                       .region_obligations(node_id)
                                       .to_vec();

        for r_o in &region_obligations {
            debug!("visit_region_obligations: r_o={}",
                   r_o.repr(self.tcx()));
            let sup_type = self.resolve_type(r_o.sup_type);
            let origin = infer::RelateParamBound(r_o.cause.span, sup_type);
            type_must_outlive(self, origin, sup_type, r_o.sub_region);
        }

        // Processing the region obligations should not cause the list to grow further:
        assert_eq!(region_obligations.len(),
                   self.fcx.inh.fulfillment_cx.borrow().region_obligations(node_id).len());
    }

    /// This method populates the region map's `free_region_map`. It walks over the transformed
    /// argument and return types for each function just before we check the body of that function,
    /// looking for types where you have a borrowed pointer to other borrowed data (e.g., `&'a &'b
    /// [uint]`.  We do not allow references to outlive the things they point at, so we can assume
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
        let tcx = self.tcx();

        for &ty in fn_sig_tys {
            let ty = self.resolve_type(ty);
            debug!("relate_free_regions(t={})", ty.repr(tcx));
            let body_scope = CodeExtent::from_node_id(body_id);
            let body_scope = ty::ReScope(body_scope);
            let implications = implicator::implications(self.fcx.infcx(), self.fcx, body_id,
                                                        ty, body_scope, span);
            for implication in implications {
                debug!("implication: {}", implication.repr(tcx));
                match implication {
                    implicator::Implication::RegionSubRegion(_,
                                                             ty::ReFree(free_a),
                                                             ty::ReFree(free_b)) => {
                        tcx.region_maps.relate_free_regions(free_a, free_b);
                    }
                    implicator::Implication::RegionSubRegion(_,
                                                             ty::ReFree(free_a),
                                                             ty::ReInfer(ty::ReVar(vid_b))) => {
                        self.fcx.inh.infcx.add_given(free_a, vid_b);
                    }
                    implicator::Implication::RegionSubRegion(..) => {
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
                    implicator::Implication::RegionSubGeneric(_, r_a, ref generic_b) => {
                        debug!("RegionSubGeneric: {} <= {}",
                               r_a.repr(tcx), generic_b.repr(tcx));

                        self.region_bound_pairs.push((r_a, generic_b.clone()));
                    }
                    implicator::Implication::Predicate(..) => { }
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

        self.fcx.infcx().resolve_regions_and_report_errors(subject_node_id);
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

    fn visit_fn(&mut self, _fk: visit::FnKind<'v>, fd: &'v ast::FnDecl,
                b: &'v ast::Block, span: Span, id: ast::NodeId) {
        self.visit_fn_body(id, fd, b, span)
    }

    fn visit_item(&mut self, i: &ast::Item) { visit_item(self, i); }

    fn visit_expr(&mut self, ex: &ast::Expr) { visit_expr(self, ex); }

    //visit_pat: visit_pat, // (..) see above

    fn visit_arm(&mut self, a: &ast::Arm) { visit_arm(self, a); }

    fn visit_local(&mut self, l: &ast::Local) { visit_local(self, l); }

    fn visit_block(&mut self, b: &ast::Block) { visit_block(self, b); }
}

fn visit_item(_rcx: &mut Rcx, _item: &ast::Item) {
    // Ignore items
}

fn visit_block(rcx: &mut Rcx, b: &ast::Block) {
    visit::walk_block(rcx, b);
}

fn visit_arm(rcx: &mut Rcx, arm: &ast::Arm) {
    // see above
    for p in &arm.pats {
        constrain_bindings_in_pat(&**p, rcx);
    }

    visit::walk_arm(rcx, arm);
}

fn visit_local(rcx: &mut Rcx, l: &ast::Local) {
    // see above
    constrain_bindings_in_pat(&*l.pat, rcx);
    link_local(rcx, l);
    visit::walk_local(rcx, l);
}

fn constrain_bindings_in_pat(pat: &ast::Pat, rcx: &mut Rcx) {
    let tcx = rcx.fcx.tcx();
    debug!("regionck::visit_pat(pat={})", pat.repr(tcx));
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

        let var_region = tcx.region_maps.var_region(id);
        type_of_node_must_outlive(
            rcx, infer::BindingTypeIsNotValidAtDecl(span),
            id, var_region);

        let var_scope = tcx.region_maps.var_scope(id);
        let typ = rcx.resolve_node_type(id);
        dropck::check_safety_of_destructor_if_necessary(rcx, typ, span, var_scope);
    })
}

fn visit_expr(rcx: &mut Rcx, expr: &ast::Expr) {
    debug!("regionck::visit_expr(e={}, repeating_scope={})",
           expr.repr(rcx.fcx.tcx()), rcx.repeating_scope);

    // No matter what, the type of each expression must outlive the
    // scope of that expression. This also guarantees basic WF.
    let expr_ty = rcx.resolve_node_type(expr.id);

    type_must_outlive(rcx, infer::ExprTypeIsNotInScope(expr_ty, expr.span),
                      expr_ty, ty::ReScope(CodeExtent::from_node_id(expr.id)));

    let method_call = MethodCall::expr(expr.id);
    let has_method_map = rcx.fcx.inh.method_map.borrow().contains_key(&method_call);

    // Check any autoderefs or autorefs that appear.
    if let Some(adjustment) = rcx.fcx.inh.adjustments.borrow().get(&expr.id) {
        debug!("adjustment={:?}", adjustment);
        match *adjustment {
            ty::AdjustDerefRef(ty::AutoDerefRef {autoderefs, autoref: ref opt_autoref}) => {
                let expr_ty = rcx.resolve_node_type(expr.id);
                constrain_autoderefs(rcx, expr, autoderefs, expr_ty);
                if let Some(ref autoref) = *opt_autoref {
                    link_autoref(rcx, expr, autoderefs, autoref);

                    // Require that the resulting region encompasses
                    // the current node.
                    //
                    // FIXME(#6268) remove to support nested method calls
                    type_of_node_must_outlive(
                        rcx, infer::AutoBorrow(expr.span),
                        expr.id, ty::ReScope(CodeExtent::from_node_id(expr.id)));
                }
            }
            /*
            ty::AutoObject(_, ref bounds, _, _) => {
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
            let mc = mc::MemCategorizationContext::new(rcx.fcx);
            mc.cat_expr_unadjusted(expr)
        };
        match cmt_result {
            Ok(head_cmt) => {
                check_safety_of_rvalue_destructor_if_necessary(rcx,
                                                               head_cmt,
                                                               expr.span);
            }
            Err(..) => {
                rcx.fcx.tcx().sess.span_note(expr.span,
                                             "cat_expr_unadjusted Errd during dtor check");
            }
        }
    }

    // If necessary, constrain destructors in this expression. This will be
    // the adjusted form if there is an adjustment.
    let cmt_result = {
        let mc = mc::MemCategorizationContext::new(rcx.fcx);
        mc.cat_expr(expr)
    };
    match cmt_result {
        Ok(head_cmt) => {
            check_safety_of_rvalue_destructor_if_necessary(rcx, head_cmt, expr.span);
        }
        Err(..) => {
            rcx.fcx.tcx().sess.span_note(expr.span,
                                         "cat_expr Errd during dtor check");
        }
    }

    match expr.node {
        ast::ExprCall(ref callee, ref args) => {
            if has_method_map {
                constrain_call(rcx, expr, Some(&**callee),
                               args.iter().map(|e| &**e), false);
            } else {
                constrain_callee(rcx, callee.id, expr, &**callee);
                constrain_call(rcx, expr, None,
                               args.iter().map(|e| &**e), false);
            }

            visit::walk_expr(rcx, expr);
        }

        ast::ExprMethodCall(_, _, ref args) => {
            constrain_call(rcx, expr, Some(&*args[0]),
                           args[1..].iter().map(|e| &**e), false);

            visit::walk_expr(rcx, expr);
        }

        ast::ExprAssignOp(_, ref lhs, ref rhs) => {
            if has_method_map {
                constrain_call(rcx, expr, Some(&**lhs),
                               Some(&**rhs).into_iter(), true);
            }

            visit::walk_expr(rcx, expr);
        }

        ast::ExprIndex(ref lhs, ref rhs) if has_method_map => {
            constrain_call(rcx, expr, Some(&**lhs),
                           Some(&**rhs).into_iter(), true);

            visit::walk_expr(rcx, expr);
        },

        ast::ExprBinary(op, ref lhs, ref rhs) if has_method_map => {
            let implicitly_ref_args = !ast_util::is_by_value_binop(op.node);

            // As `expr_method_call`, but the call is via an
            // overloaded op.  Note that we (sadly) currently use an
            // implicit "by ref" sort of passing style here.  This
            // should be converted to an adjustment!
            constrain_call(rcx, expr, Some(&**lhs),
                           Some(&**rhs).into_iter(), implicitly_ref_args);

            visit::walk_expr(rcx, expr);
        }

        ast::ExprBinary(_, ref lhs, ref rhs) => {
            // If you do `x OP y`, then the types of `x` and `y` must
            // outlive the operation you are performing.
            let lhs_ty = rcx.resolve_expr_type_adjusted(&**lhs);
            let rhs_ty = rcx.resolve_expr_type_adjusted(&**rhs);
            for &ty in [lhs_ty, rhs_ty].iter() {
                type_must_outlive(rcx,
                                  infer::Operand(expr.span),
                                  ty,
                                  ty::ReScope(CodeExtent::from_node_id(expr.id)));
            }
            visit::walk_expr(rcx, expr);
        }

        ast::ExprUnary(op, ref lhs) if has_method_map => {
            let implicitly_ref_args = !ast_util::is_by_value_unop(op);

            // As above.
            constrain_call(rcx, expr, Some(&**lhs),
                           None::<ast::Expr>.iter(), implicitly_ref_args);

            visit::walk_expr(rcx, expr);
        }

        ast::ExprUnary(ast::UnDeref, ref base) => {
            // For *a, the lifetime of a must enclose the deref
            let method_call = MethodCall::expr(expr.id);
            let base_ty = match rcx.fcx.inh.method_map.borrow().get(&method_call) {
                Some(method) => {
                    constrain_call(rcx, expr, Some(&**base),
                                   None::<ast::Expr>.iter(), true);
                    let fn_ret = // late-bound regions in overloaded method calls are instantiated
                        ty::no_late_bound_regions(rcx.tcx(), &ty::ty_fn_ret(method.ty)).unwrap();
                    fn_ret.unwrap()
                }
                None => rcx.resolve_node_type(base.id)
            };
            if let ty::ty_rptr(r_ptr, _) = base_ty.sty {
                mk_subregion_due_to_dereference(
                    rcx, expr.span, ty::ReScope(CodeExtent::from_node_id(expr.id)), *r_ptr);
            }

            visit::walk_expr(rcx, expr);
        }

        ast::ExprIndex(ref vec_expr, _) => {
            // For a[b], the lifetime of a must enclose the deref
            let vec_type = rcx.resolve_expr_type_adjusted(&**vec_expr);
            constrain_index(rcx, expr, vec_type);

            visit::walk_expr(rcx, expr);
        }

        ast::ExprCast(ref source, _) => {
            // Determine if we are casting `source` to a trait
            // instance.  If so, we have to be sure that the type of
            // the source obeys the trait's region bound.
            constrain_cast(rcx, expr, &**source);
            visit::walk_expr(rcx, expr);
        }

        ast::ExprAddrOf(m, ref base) => {
            link_addr_of(rcx, expr, m, &**base);

            // Require that when you write a `&expr` expression, the
            // resulting pointer has a lifetime that encompasses the
            // `&expr` expression itself. Note that we constraining
            // the type of the node expr.id here *before applying
            // adjustments*.
            //
            // FIXME(#6268) nested method calls requires that this rule change
            let ty0 = rcx.resolve_node_type(expr.id);
            type_must_outlive(rcx, infer::AddrOf(expr.span),
                              ty0, ty::ReScope(CodeExtent::from_node_id(expr.id)));
            visit::walk_expr(rcx, expr);
        }

        ast::ExprMatch(ref discr, ref arms, _) => {
            link_match(rcx, &**discr, &arms[..]);

            visit::walk_expr(rcx, expr);
        }

        ast::ExprClosure(_, _, ref body) => {
            check_expr_fn_block(rcx, expr, &**body);
        }

        ast::ExprLoop(ref body, _) => {
            let repeating_scope = rcx.set_repeating_scope(body.id);
            visit::walk_expr(rcx, expr);
            rcx.set_repeating_scope(repeating_scope);
        }

        ast::ExprWhile(ref cond, ref body, _) => {
            let repeating_scope = rcx.set_repeating_scope(cond.id);
            rcx.visit_expr(&**cond);

            rcx.set_repeating_scope(body.id);
            rcx.visit_block(&**body);

            rcx.set_repeating_scope(repeating_scope);
        }

        _ => {
            visit::walk_expr(rcx, expr);
        }
    }
}

fn constrain_cast(rcx: &mut Rcx,
                  cast_expr: &ast::Expr,
                  source_expr: &ast::Expr)
{
    debug!("constrain_cast(cast_expr={}, source_expr={})",
           cast_expr.repr(rcx.tcx()),
           source_expr.repr(rcx.tcx()));

    let source_ty = rcx.resolve_node_type(source_expr.id);
    let target_ty = rcx.resolve_node_type(cast_expr.id);

    walk_cast(rcx, cast_expr, source_ty, target_ty);

    fn walk_cast<'a, 'tcx>(rcx: &mut Rcx<'a, 'tcx>,
                           cast_expr: &ast::Expr,
                           from_ty: Ty<'tcx>,
                           to_ty: Ty<'tcx>) {
        debug!("walk_cast(from_ty={}, to_ty={})",
               from_ty.repr(rcx.tcx()),
               to_ty.repr(rcx.tcx()));
        match (&from_ty.sty, &to_ty.sty) {
            /*From:*/ (&ty::ty_rptr(from_r, ref from_mt),
            /*To:  */  &ty::ty_rptr(to_r, ref to_mt)) => {
                // Target cannot outlive source, naturally.
                rcx.fcx.mk_subr(infer::Reborrow(cast_expr.span), *to_r, *from_r);
                walk_cast(rcx, cast_expr, from_mt.ty, to_mt.ty);
            }

            /*From:*/ (_,
            /*To:  */  &ty::ty_trait(box ty::TyTrait { ref bounds, .. })) => {
                // When T is existentially quantified as a trait
                // `Foo+'to`, it must outlive the region bound `'to`.
                type_must_outlive(rcx, infer::RelateObjectBound(cast_expr.span),
                                  from_ty, bounds.region_bound);
            }

            /*From:*/ (&ty::ty_uniq(from_referent_ty),
            /*To:  */  &ty::ty_uniq(to_referent_ty)) => {
                walk_cast(rcx, cast_expr, from_referent_ty, to_referent_ty);
            }

            _ => { }
        }
    }
}

fn check_expr_fn_block(rcx: &mut Rcx,
                       expr: &ast::Expr,
                       body: &ast::Block) {
    let tcx = rcx.fcx.tcx();
    let function_type = rcx.resolve_node_type(expr.id);

    match function_type.sty {
        ty::ty_closure(_, region, _) => {
            ty::with_freevars(tcx, expr.id, |freevars| {
                constrain_captured_variables(rcx, *region, expr, freevars);
            })
        }
        _ => { }
    }

    let repeating_scope = rcx.set_repeating_scope(body.id);
    visit::walk_expr(rcx, expr);
    rcx.set_repeating_scope(repeating_scope);

    match function_type.sty {
        ty::ty_closure(_, region, _) => {
            ty::with_freevars(tcx, expr.id, |freevars| {
                let bounds = ty::region_existential_bound(*region);
                ensure_free_variable_types_outlive_closure_bound(rcx, &bounds, expr, freevars);
            })
        }
        _ => {}
    }

    /// Make sure that the type of all free variables referenced inside a closure/proc outlive the
    /// closure/proc's lifetime bound. This is just a special case of the usual rules about closed
    /// over values outliving the object's lifetime bound.
    fn ensure_free_variable_types_outlive_closure_bound(
        rcx: &mut Rcx,
        bounds: &ty::ExistentialBounds,
        expr: &ast::Expr,
        freevars: &[ty::Freevar])
    {
        let tcx = rcx.fcx.ccx.tcx;

        debug!("ensure_free_variable_types_outlive_closure_bound({}, {})",
               bounds.region_bound.repr(tcx), expr.repr(tcx));

        for freevar in freevars {
            let var_node_id = {
                let def_id = freevar.def.def_id();
                assert!(def_id.krate == ast::LOCAL_CRATE);
                def_id.node
            };

            // Compute the type of the field in the environment that
            // represents `var_node_id`.  For a by-value closure, this
            // will be the same as the type of the variable.  For a
            // by-reference closure, this will be `&T` where `T` is
            // the type of the variable.
            let raw_var_ty = rcx.resolve_node_type(var_node_id);
            let upvar_id = ty::UpvarId { var_id: var_node_id,
                                         closure_expr_id: expr.id };
            let var_ty = match rcx.fcx.inh.upvar_capture_map.borrow()[upvar_id] {
                ty::UpvarCapture::ByRef(ref upvar_borrow) => {
                    ty::mk_rptr(rcx.tcx(),
                                rcx.tcx().mk_region(upvar_borrow.region),
                                ty::mt { mutbl: upvar_borrow.kind.to_mutbl_lossy(),
                                         ty: raw_var_ty })
                }
                ty::UpvarCapture::ByValue => raw_var_ty,
            };

            // Check that the type meets the criteria of the existential bounds:
            for builtin_bound in &bounds.builtin_bounds {
                let code = traits::ClosureCapture(var_node_id, expr.span, builtin_bound);
                let cause = traits::ObligationCause::new(freevar.span, rcx.fcx.body_id, code);
                rcx.fcx.register_builtin_bound(var_ty, builtin_bound, cause);
            }

            type_must_outlive(
                rcx, infer::FreeVariable(expr.span, var_node_id),
                var_ty, bounds.region_bound);
        }
    }

    /// Make sure that all free variables referenced inside the closure outlive the closure's
    /// lifetime bound. Also, create an entry in the upvar_borrows map with a region.
    fn constrain_captured_variables(
        rcx: &mut Rcx,
        region_bound: ty::Region,
        expr: &ast::Expr,
        freevars: &[ty::Freevar])
    {
        let tcx = rcx.fcx.ccx.tcx;
        debug!("constrain_captured_variables({}, {})",
               region_bound.repr(tcx), expr.repr(tcx));
        for freevar in freevars {
            debug!("constrain_captured_variables: freevar.def={:?}", freevar.def);

            // Identify the variable being closed over and its node-id.
            let def = freevar.def;
            let var_node_id = {
                let def_id = def.def_id();
                assert!(def_id.krate == ast::LOCAL_CRATE);
                def_id.node
            };
            let upvar_id = ty::UpvarId { var_id: var_node_id,
                                         closure_expr_id: expr.id };

            match rcx.fcx.inh.upvar_capture_map.borrow()[upvar_id] {
                ty::UpvarCapture::ByValue => { }
                ty::UpvarCapture::ByRef(upvar_borrow) => {
                    rcx.fcx.mk_subr(infer::FreeVariable(freevar.span, var_node_id),
                                    region_bound, upvar_borrow.region);

                    // Guarantee that the closure does not outlive the variable itself.
                    let enclosing_region = region_of_def(rcx.fcx, def);
                    debug!("constrain_captured_variables: enclosing_region = {}",
                           enclosing_region.repr(tcx));
                    rcx.fcx.mk_subr(infer::FreeVariable(freevar.span, var_node_id),
                                    region_bound, enclosing_region);
                }
            }
        }
    }
}

fn constrain_callee(rcx: &mut Rcx,
                    callee_id: ast::NodeId,
                    _call_expr: &ast::Expr,
                    _callee_expr: &ast::Expr) {
    let callee_ty = rcx.resolve_node_type(callee_id);
    match callee_ty.sty {
        ty::ty_bare_fn(..) => { }
        _ => {
            // this should not happen, but it does if the program is
            // erroneous
            //
            // tcx.sess.span_bug(
            //     callee_expr.span,
            //     format!("Calling non-function: {}", callee_ty.repr(tcx)));
        }
    }
}

fn constrain_call<'a, I: Iterator<Item=&'a ast::Expr>>(rcx: &mut Rcx,
                                                       call_expr: &ast::Expr,
                                                       receiver: Option<&ast::Expr>,
                                                       arg_exprs: I,
                                                       implicitly_ref_args: bool) {
    //! Invoked on every call site (i.e., normal calls, method calls,
    //! and overloaded operators). Constrains the regions which appear
    //! in the type of the function. Also constrains the regions that
    //! appear in the arguments appropriately.

    let tcx = rcx.fcx.tcx();
    debug!("constrain_call(call_expr={}, \
            receiver={}, \
            implicitly_ref_args={})",
            call_expr.repr(tcx),
            receiver.repr(tcx),
            implicitly_ref_args);

    // `callee_region` is the scope representing the time in which the
    // call occurs.
    //
    // FIXME(#6268) to support nested method calls, should be callee_id
    let callee_scope = CodeExtent::from_node_id(call_expr.id);
    let callee_region = ty::ReScope(callee_scope);

    debug!("callee_region={}", callee_region.repr(tcx));

    for arg_expr in arg_exprs {
        debug!("Argument: {}", arg_expr.repr(tcx));

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
        debug!("receiver: {}", r.repr(tcx));
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
                                  deref_expr: &ast::Expr,
                                  derefs: uint,
                                  mut derefd_ty: Ty<'tcx>)
{
    debug!("constrain_autoderefs(deref_expr={}, derefs={}, derefd_ty={})",
           deref_expr.repr(rcx.tcx()),
           derefs,
           derefd_ty.repr(rcx.tcx()));

    let r_deref_expr = ty::ReScope(CodeExtent::from_node_id(deref_expr.id));
    for i in 0..derefs {
        let method_call = MethodCall::autoderef(deref_expr.id, i);
        debug!("constrain_autoderefs: method_call={:?} (of {:?} total)", method_call, derefs);

        derefd_ty = match rcx.fcx.inh.method_map.borrow().get(&method_call) {
            Some(method) => {
                debug!("constrain_autoderefs: #{} is overloaded, method={}",
                       i, method.repr(rcx.tcx()));

                // Treat overloaded autoderefs as if an AutoRef adjustment
                // was applied on the base type, as that is always the case.
                let fn_sig = ty::ty_fn_sig(method.ty);
                let fn_sig = // late-bound regions should have been instantiated
                    ty::no_late_bound_regions(rcx.tcx(), fn_sig).unwrap();
                let self_ty = fn_sig.inputs[0];
                let (m, r) = match self_ty.sty {
                    ty::ty_rptr(r, ref m) => (m.mutbl, r),
                    _ => {
                        rcx.tcx().sess.span_bug(
                            deref_expr.span,
                            &format!("bad overloaded deref type {}",
                                     method.ty.repr(rcx.tcx()))[])
                    }
                };

                debug!("constrain_autoderefs: receiver r={:?} m={:?}",
                       r.repr(rcx.tcx()), m);

                {
                    let mc = mc::MemCategorizationContext::new(rcx.fcx);
                    let self_cmt = ignore_err!(mc.cat_expr_autoderefd(deref_expr, i));
                    debug!("constrain_autoderefs: self_cmt={:?}",
                           self_cmt.repr(rcx.tcx()));
                    link_region(rcx, deref_expr.span, *r,
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

        if let ty::ty_rptr(r_ptr, _) =  derefd_ty.sty {
            mk_subregion_due_to_dereference(rcx, deref_expr.span,
                                            r_deref_expr, *r_ptr);
        }

        match ty::deref(derefd_ty, true) {
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
        mc::cat_rvalue(region) => {
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
                                 format!("unexpected rvalue region in rvalue \
                                          destructor safety checking: `{}`",
                                         region.repr(rcx.tcx())).as_slice());
                }
            }
        }
        _ => {}
    }
}

/// Invoked on any index expression that occurs. Checks that if this is a slice being indexed, the
/// lifetime of the pointer includes the deref expr.
fn constrain_index<'a, 'tcx>(rcx: &mut Rcx<'a, 'tcx>,
                             index_expr: &ast::Expr,
                             indexed_ty: Ty<'tcx>)
{
    debug!("constrain_index(index_expr=?, indexed_ty={}",
           rcx.fcx.infcx().ty_to_string(indexed_ty));

    let r_index_expr = ty::ReScope(CodeExtent::from_node_id(index_expr.id));
    if let ty::ty_rptr(r_ptr, mt) = indexed_ty.sty {
        match mt.ty.sty {
            ty::ty_vec(_, None) | ty::ty_str => {
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
    let ty = ty::adjust_ty(tcx, origin.span(), id, ty0,
                           rcx.fcx.inh.adjustments.borrow().get(&id),
                           |method_call| rcx.resolve_method_type(method_call));
    debug!("constrain_regions_in_type_of_node(\
            ty={}, ty0={}, id={}, minimum_lifetime={:?})",
           ty_to_string(tcx, ty), ty_to_string(tcx, ty0),
           id, minimum_lifetime);
    type_must_outlive(rcx, origin, ty, minimum_lifetime);
}

/// Computes the guarantor for an expression `&base` and then ensures that the lifetime of the
/// resulting pointer is linked to the lifetime of its guarantor (if any).
fn link_addr_of(rcx: &mut Rcx, expr: &ast::Expr,
                mutability: ast::Mutability, base: &ast::Expr) {
    debug!("link_addr_of(expr={}, base={})", expr.repr(rcx.tcx()), base.repr(rcx.tcx()));

    let cmt = {
        let mc = mc::MemCategorizationContext::new(rcx.fcx);
        ignore_err!(mc.cat_expr(base))
    };

    debug!("link_addr_of: cmt={}", cmt.repr(rcx.tcx()));

    link_region_from_node_type(rcx, expr.span, expr.id, mutability, cmt);
}

/// Computes the guarantors for any ref bindings in a `let` and
/// then ensures that the lifetime of the resulting pointer is
/// linked to the lifetime of the initialization expression.
fn link_local(rcx: &Rcx, local: &ast::Local) {
    debug!("regionck::for_local()");
    let init_expr = match local.init {
        None => { return; }
        Some(ref expr) => &**expr,
    };
    let mc = mc::MemCategorizationContext::new(rcx.fcx);
    let discr_cmt = ignore_err!(mc.cat_expr(init_expr));
    link_pattern(rcx, mc, discr_cmt, &*local.pat);
}

/// Computes the guarantors for any ref bindings in a match and
/// then ensures that the lifetime of the resulting pointer is
/// linked to the lifetime of its guarantor (if any).
fn link_match(rcx: &Rcx, discr: &ast::Expr, arms: &[ast::Arm]) {
    debug!("regionck::for_match()");
    let mc = mc::MemCategorizationContext::new(rcx.fcx);
    let discr_cmt = ignore_err!(mc.cat_expr(discr));
    debug!("discr_cmt={}", discr_cmt.repr(rcx.tcx()));
    for arm in arms {
        for root_pat in &arm.pats {
            link_pattern(rcx, mc, discr_cmt.clone(), &**root_pat);
        }
    }
}

/// Computes the guarantors for any ref bindings in a match and
/// then ensures that the lifetime of the resulting pointer is
/// linked to the lifetime of its guarantor (if any).
fn link_fn_args(rcx: &Rcx, body_scope: CodeExtent, args: &[ast::Arg]) {
    debug!("regionck::link_fn_args(body_scope={:?})", body_scope);
    let mc = mc::MemCategorizationContext::new(rcx.fcx);
    for arg in args {
        let arg_ty = rcx.fcx.node_ty(arg.id);
        let re_scope = ty::ReScope(body_scope);
        let arg_cmt = mc.cat_rvalue(arg.id, arg.ty.span, re_scope, arg_ty);
        debug!("arg_ty={} arg_cmt={}",
               arg_ty.repr(rcx.tcx()),
               arg_cmt.repr(rcx.tcx()));
        link_pattern(rcx, mc, arg_cmt, &*arg.pat);
    }
}

/// Link lifetimes of any ref bindings in `root_pat` to the pointers found in the discriminant, if
/// needed.
fn link_pattern<'a, 'tcx>(rcx: &Rcx<'a, 'tcx>,
                          mc: mc::MemCategorizationContext<FnCtxt<'a, 'tcx>>,
                          discr_cmt: mc::cmt<'tcx>,
                          root_pat: &ast::Pat) {
    debug!("link_pattern(discr_cmt={}, root_pat={})",
           discr_cmt.repr(rcx.tcx()),
           root_pat.repr(rcx.tcx()));
    let _ = mc.cat_pattern(discr_cmt, root_pat, |mc, sub_cmt, sub_pat| {
            match sub_pat.node {
                // `ref x` pattern
                ast::PatIdent(ast::BindByRef(mutbl), _, _) => {
                    link_region_from_node_type(
                        rcx, sub_pat.span, sub_pat.id,
                        mutbl, sub_cmt);
                }

                // `[_, ..slice, _]` pattern
                ast::PatVec(_, Some(ref slice_pat), _) => {
                    match mc.cat_slice_pattern(sub_cmt, &**slice_pat) {
                        Ok((slice_cmt, slice_mutbl, slice_r)) => {
                            link_region(rcx, sub_pat.span, slice_r,
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
                expr: &ast::Expr,
                autoderefs: uint,
                autoref: &ty::AutoRef) {

    debug!("link_autoref(autoref={:?})", autoref);
    let mc = mc::MemCategorizationContext::new(rcx.fcx);
    let expr_cmt = ignore_err!(mc.cat_expr_autoderefd(expr, autoderefs));
    debug!("expr_cmt={}", expr_cmt.repr(rcx.tcx()));

    match *autoref {
        ty::AutoPtr(r, m, _) => {
            link_region(rcx, expr.span, r,
                ty::BorrowKind::from_mutbl(m), expr_cmt);
        }

        ty::AutoUnsafe(..) | ty::AutoUnsizeUniq(_) | ty::AutoUnsize(_) => {}
    }
}

/// Computes the guarantor for cases where the `expr` is being passed by implicit reference and
/// must outlive `callee_scope`.
fn link_by_ref(rcx: &Rcx,
               expr: &ast::Expr,
               callee_scope: CodeExtent) {
    let tcx = rcx.tcx();
    debug!("link_by_ref(expr={}, callee_scope={:?})",
           expr.repr(tcx), callee_scope);
    let mc = mc::MemCategorizationContext::new(rcx.fcx);
    let expr_cmt = ignore_err!(mc.cat_expr(expr));
    let borrow_region = ty::ReScope(callee_scope);
    link_region(rcx, expr.span, borrow_region, ty::ImmBorrow, expr_cmt);
}

/// Like `link_region()`, except that the region is extracted from the type of `id`, which must be
/// some reference (`&T`, `&str`, etc).
fn link_region_from_node_type<'a, 'tcx>(rcx: &Rcx<'a, 'tcx>,
                                        span: Span,
                                        id: ast::NodeId,
                                        mutbl: ast::Mutability,
                                        cmt_borrowed: mc::cmt<'tcx>) {
    debug!("link_region_from_node_type(id={:?}, mutbl={:?}, cmt_borrowed={})",
           id, mutbl, cmt_borrowed.repr(rcx.tcx()));

    let rptr_ty = rcx.resolve_node_type(id);
    if !ty::type_is_error(rptr_ty) {
        let tcx = rcx.fcx.ccx.tcx;
        debug!("rptr_ty={}", ty_to_string(tcx, rptr_ty));
        let r = ty::ty_region(tcx, span, rptr_ty);
        link_region(rcx, span, r, ty::BorrowKind::from_mutbl(mutbl),
                    cmt_borrowed);
    }
}

/// Informs the inference engine that `borrow_cmt` is being borrowed with kind `borrow_kind` and
/// lifetime `borrow_region`. In order to ensure borrowck is satisfied, this may create constraints
/// between regions, as explained in `link_reborrowed_region()`.
fn link_region<'a, 'tcx>(rcx: &Rcx<'a, 'tcx>,
                         span: Span,
                         borrow_region: ty::Region,
                         borrow_kind: ty::BorrowKind,
                         borrow_cmt: mc::cmt<'tcx>) {
    let mut borrow_cmt = borrow_cmt;
    let mut borrow_kind = borrow_kind;

    loop {
        debug!("link_region(borrow_region={}, borrow_kind={}, borrow_cmt={})",
               borrow_region.repr(rcx.tcx()),
               borrow_kind.repr(rcx.tcx()),
               borrow_cmt.repr(rcx.tcx()));
        match borrow_cmt.cat.clone() {
            mc::cat_deref(ref_cmt, _,
                          mc::Implicit(ref_kind, ref_region)) |
            mc::cat_deref(ref_cmt, _,
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

            mc::cat_downcast(cmt_base, _) |
            mc::cat_deref(cmt_base, _, mc::Unique) |
            mc::cat_interior(cmt_base, _) => {
                // Borrowing interior or owned data requires the base
                // to be valid and borrowable in the same fashion.
                borrow_cmt = cmt_base;
                borrow_kind = borrow_kind;
            }

            mc::cat_deref(_, _, mc::UnsafePtr(..)) |
            mc::cat_static_item |
            mc::cat_upvar(..) |
            mc::cat_local(..) |
            mc::cat_rvalue(..) => {
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
                                    borrow_region: ty::Region,
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
            let upvar_capture_map = rcx.fcx.inh.upvar_capture_map.borrow_mut();
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
                        &format!("Illegal upvar id: {}",
                                upvar_id.repr(rcx.tcx()))[]);
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

    debug!("link_reborrowed_region: {} <= {}",
           borrow_region.repr(rcx.tcx()),
           ref_region.repr(rcx.tcx()));
    rcx.fcx.mk_subr(cause, borrow_region, ref_region);

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

/// Ensures that all borrowed data reachable via `ty` outlives `region`.
pub fn type_must_outlive<'a, 'tcx>(rcx: &mut Rcx<'a, 'tcx>,
                               origin: infer::SubregionOrigin<'tcx>,
                               ty: Ty<'tcx>,
                               region: ty::Region)
{
    debug!("type_must_outlive(ty={}, region={})",
           ty.repr(rcx.tcx()),
           region.repr(rcx.tcx()));

    let implications = implicator::implications(rcx.fcx.infcx(), rcx.fcx, rcx.body_id,
                                                ty, region, origin.span());
    for implication in implications {
        debug!("implication: {}", implication.repr(rcx.tcx()));
        match implication {
            implicator::Implication::RegionSubRegion(None, r_a, r_b) => {
                rcx.fcx.mk_subr(origin.clone(), r_a, r_b);
            }
            implicator::Implication::RegionSubRegion(Some(ty), r_a, r_b) => {
                let o1 = infer::ReferenceOutlivesReferent(ty, origin.span());
                rcx.fcx.mk_subr(o1, r_a, r_b);
            }
            implicator::Implication::RegionSubGeneric(None, r_a, ref generic_b) => {
                generic_must_outlive(rcx, origin.clone(), r_a, generic_b);
            }
            implicator::Implication::RegionSubGeneric(Some(ty), r_a, ref generic_b) => {
                let o1 = infer::ReferenceOutlivesReferent(ty, origin.span());
                generic_must_outlive(rcx, o1, r_a, generic_b);
            }
            implicator::Implication::Predicate(def_id, predicate) => {
                let cause = traits::ObligationCause::new(origin.span(),
                                                         rcx.body_id,
                                                         traits::ItemObligation(def_id));
                let obligation = traits::Obligation::new(cause, predicate);
                rcx.fcx.register_predicate(obligation);
            }
        }
    }
}

fn generic_must_outlive<'a, 'tcx>(rcx: &Rcx<'a, 'tcx>,
                                  origin: infer::SubregionOrigin<'tcx>,
                                  region: ty::Region,
                                  generic: &GenericKind<'tcx>) {
    let param_env = &rcx.fcx.inh.param_env;

    debug!("param_must_outlive(region={}, generic={})",
           region.repr(rcx.tcx()),
           generic.repr(rcx.tcx()));

    // To start, collect bounds from user:
    let mut param_bounds =
        ty::required_region_bounds(rcx.tcx(),
                                   generic.to_ty(rcx.tcx()),
                                   param_env.caller_bounds.clone());

    // In the case of a projection T::Foo, we may be able to extract bounds from the trait def:
    match *generic {
        GenericKind::Param(..) => { }
        GenericKind::Projection(ref projection_ty) => {
            param_bounds.push_all(
                &projection_bounds(rcx, origin.span(), projection_ty)[]);
        }
    }

    // Add in the default bound of fn body that applies to all in
    // scope type parameters:
    param_bounds.push(param_env.implicit_region_bound);

    // Finally, collect regions we scraped from the well-formedness
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
    for &(ref r, ref p) in &rcx.region_bound_pairs {
        debug!("generic={} p={}",
               generic.repr(rcx.tcx()),
               p.repr(rcx.tcx()));
        if generic == p {
            param_bounds.push(*r);
        }
    }

    // Inform region inference that this generic must be properly
    // bounded.
    rcx.fcx.infcx().verify_generic_bound(origin,
                                         generic.clone(),
                                         region,
                                         param_bounds);
}

fn projection_bounds<'a,'tcx>(rcx: &Rcx<'a, 'tcx>,
                              span: Span,
                              projection_ty: &ty::ProjectionTy<'tcx>)
                              -> Vec<ty::Region>
{
    let fcx = rcx.fcx;
    let tcx = fcx.tcx();
    let infcx = fcx.infcx();

    debug!("projection_bounds(projection_ty={})",
           projection_ty.repr(tcx));

    let ty = ty::mk_projection(tcx, projection_ty.trait_ref.clone(), projection_ty.item_name);

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
    let trait_predicates = ty::lookup_predicates(tcx, projection_ty.trait_ref.def_id);
    let predicates = trait_predicates.predicates.as_slice().to_vec();
    traits::elaborate_predicates(tcx, predicates)
        .filter_map(|predicate| {
            // we're only interesting in `T : 'a` style predicates:
            let outlives = match predicate {
                ty::Predicate::TypeOutlives(data) => data,
                _ => { return None; }
            };

            debug!("projection_bounds: outlives={} (1)",
                   outlives.repr(tcx));

            // apply the substitutions (and normalize any projected types)
            let outlives = fcx.instantiate_type_scheme(span,
                                                       projection_ty.trait_ref.substs,
                                                       &outlives);

            debug!("projection_bounds: outlives={} (2)",
                   outlives.repr(tcx));

            let region_result = infcx.try(|_| {
                let (outlives, _) =
                    infcx.replace_late_bound_regions_with_fresh_var(
                        span,
                        infer::AssocTypeProjection(projection_ty.item_name),
                        &outlives);

                debug!("projection_bounds: outlives={} (3)",
                       outlives.repr(tcx));

                // check whether this predicate applies to our current projection
                match infer::mk_eqty(infcx, false, infer::Misc(span), ty, outlives.0) {
                    Ok(()) => { Ok(outlives.1) }
                    Err(_) => { Err(()) }
                }
            });

            debug!("projection_bounds: region_result={}",
                   region_result.repr(tcx));

            region_result.ok()
        })
        .collect()
}
