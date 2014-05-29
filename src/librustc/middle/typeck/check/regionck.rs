// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

The region check is a final pass that runs over the AST after we have
inferred the type constraints but before we have actually finalized
the types.  Its purpose is to embed a variety of region constraints.
Inserting these constraints as a separate pass is good because (1) it
localizes the code that has to do with region inference and (2) often
we cannot know what constraints are needed until the basic types have
been inferred.

### Interaction with the borrow checker

In general, the job of the borrowck module (which runs later) is to
check that all soundness criteria are met, given a particular set of
regions. The job of *this* module is to anticipate the needs of the
borrow checker and infer regions that will satisfy its requirements.
It is generally true that the inference doesn't need to be sound,
meaning that if there is a bug and we inferred bad regions, the borrow
checker should catch it. This is not entirely true though; for
example, the borrow checker doesn't check subtyping, and it doesn't
check that region pointers are always live when they are used. It
might be worthwhile to fix this so that borrowck serves as a kind of
verification step -- that would add confidence in the overall
correctness of the compiler, at the cost of duplicating some type
checks and effort.

### Inferring the duration of borrows, automatic and otherwise

Whenever we introduce a borrowed pointer, for example as the result of
a borrow expression `let x = &data`, the lifetime of the pointer `x`
is always specified as a region inference variable. `regionck` has the
job of adding constraints such that this inference variable is as
narrow as possible while still accommodating all uses (that is, every
dereference of the resulting pointer must be within the lifetime).

#### Reborrows

Generally speaking, `regionck` does NOT try to ensure that the data
`data` will outlive the pointer `x`. That is the job of borrowck.  The
one exception is when "re-borrowing" the contents of another borrowed
pointer. For example, imagine you have a borrowed pointer `b` with
lifetime L1 and you have an expression `&*b`. The result of this
expression will be another borrowed pointer with lifetime L2 (which is
an inference variable). The borrow checker is going to enforce the
constraint that L2 < L1, because otherwise you are re-borrowing data
for a lifetime larger than the original loan.  However, without the
routines in this module, the region inferencer would not know of this
dependency and thus it might infer the lifetime of L2 to be greater
than L1 (issue #3148).

There are a number of troublesome scenarios in the tests
`region-dependent-*.rs`, but here is one example:

    struct Foo { i: int }
    struct Bar { foo: Foo  }
    fn get_i(x: &'a Bar) -> &'a int {
       let foo = &x.foo; // Lifetime L1
       &foo.i            // Lifetime L2
    }

Note that this comes up either with `&` expressions, `ref`
bindings, and `autorefs`, which are the three ways to introduce
a borrow.

The key point here is that when you are borrowing a value that
is "guaranteed" by a borrowed pointer, you must link the
lifetime of that borrowed pointer (L1, here) to the lifetime of
the borrow itself (L2).  What do I mean by "guaranteed" by a
borrowed pointer? I mean any data that is reached by first
dereferencing a borrowed pointer and then either traversing
interior offsets or owned pointers.  We say that the guarantor
of such data it the region of the borrowed pointer that was
traversed.  This is essentially the same as the ownership
relation, except that a borrowed pointer never owns its
contents.

### Inferring borrow kinds for upvars

Whenever there is a closure expression, we need to determine how each
upvar is used. We do this by initially assigning each upvar an
immutable "borrow kind" (see `ty::BorrowKind` for details) and then
"escalating" the kind as needed. The borrow kind proceeds according to
the following lattice:

    ty::ImmBorrow -> ty::UniqueImmBorrow -> ty::MutBorrow

So, for example, if we see an assignment `x = 5` to an upvar `x`, we
will promote its borrow kind to mutable borrow. If we see an `&mut x`
we'll do the same. Naturally, this applies not just to the upvar, but
to everything owned by `x`, so the result is the same for something
like `x.f = 5` and so on (presuming `x` is not a borrowed pointer to a
struct). These adjustments are performed in
`adjust_upvar_borrow_kind()` (you can trace backwards through the code
from there).

The fact that we are inferring borrow kinds as we go results in a
semi-hacky interaction with mem-categorization. In particular,
mem-categorization will query the current borrow kind as it
categorizes, and we'll return the *current* value, but this may get
adjusted later. Therefore, in this module, we generally ignore the
borrow kind (and derived mutabilities) that are returned from
mem-categorization, since they may be inaccurate. (Another option
would be to use a unification scheme, where instead of returning a
concrete borrow kind like `ty::ImmBorrow`, we return a
`ty::InferBorrow(upvar_id)` or something like that, but this would
then mean that all later passes would have to check for these figments
and report an error, and it just seems like more mess in the end.)

*/


use middle::freevars;
use mc = middle::mem_categorization;
use middle::ty::{ReScope};
use middle::ty;
use middle::typeck::astconv::AstConv;
use middle::typeck::check::FnCtxt;
use middle::typeck::check::regionmanip::relate_nested_regions;
use middle::typeck::infer::resolve_and_force_all_but_regions;
use middle::typeck::infer::resolve_type;
use middle::typeck::infer;
use middle::typeck::MethodCall;
use middle::pat_util;
use util::nodemap::NodeMap;
use util::ppaux::{ty_to_str, region_to_str, Repr};

use syntax::ast;
use syntax::codemap::Span;
use syntax::visit;
use syntax::visit::Visitor;

use std::cell::RefCell;

// If mem categorization results in an error, it's because the type
// check failed (or will fail, when the error is uncovered and
// reported during writeback). In this case, we just ignore this part
// of the code and don't try to add any more region constraints.
macro_rules! ignore_err(
    ($inp: expr) => (
        match $inp {
            Ok(v) => v,
            Err(()) => return
        }
    )
)

pub struct Rcx<'a> {
    fcx: &'a FnCtxt<'a>,

    // id of innermost fn or loop
    repeating_scope: ast::NodeId,
}

impl<'a> Rcx<'a> {
    pub fn tcx(&self) -> &'a ty::ctxt {
        self.fcx.ccx.tcx
    }

    pub fn set_repeating_scope(&mut self, scope: ast::NodeId) -> ast::NodeId {
        let old_scope = self.repeating_scope;
        self.repeating_scope = scope;
        old_scope
    }

    pub fn resolve_type(&self, unresolved_ty: ty::t) -> ty::t {
        /*!
         * Try to resolve the type for the given node, returning
         * t_err if an error results.  Note that we never care
         * about the details of the error, the same error will be
         * detected and reported in the writeback phase.
         *
         * Note one important point: we do not attempt to resolve
         * *region variables* here.  This is because regionck is
         * essentially adding constraints to those region variables
         * and so may yet influence how they are resolved.
         *
         * Consider this silly example:
         *
         *     fn borrow(x: &int) -> &int {x}
         *     fn foo(x: @int) -> int {  // block: B
         *         let b = borrow(x);    // region: <R0>
         *         *b
         *     }
         *
         * Here, the region of `b` will be `<R0>`.  `<R0>` is
         * constrainted to be some subregion of the block B and some
         * superregion of the call.  If we forced it now, we'd choose
         * the smaller region (the call).  But that would make the *b
         * illegal.  Since we don't resolve, the type of b will be
         * `&<R0>.int` and then `*b` will require that `<R0>` be
         * bigger than the let and the `*b` expression, so we will
         * effectively resolve `<R0>` to be the block B.
         */
        match resolve_type(self.fcx.infcx(), unresolved_ty,
                           resolve_and_force_all_but_regions) {
            Ok(t) => t,
            Err(_) => ty::mk_err()
        }
    }

    /// Try to resolve the type for the given node.
    fn resolve_node_type(&self, id: ast::NodeId) -> ty::t {
        let t = self.fcx.node_ty(id);
        self.resolve_type(t)
    }

    fn resolve_method_type(&self, method_call: MethodCall) -> Option<ty::t> {
        let method_ty = self.fcx.inh.method_map.borrow()
                            .find(&method_call).map(|method| method.ty);
        method_ty.map(|method_ty| self.resolve_type(method_ty))
    }

    /// Try to resolve the type for the given node.
    pub fn resolve_expr_type_adjusted(&mut self, expr: &ast::Expr) -> ty::t {
        let ty_unadjusted = self.resolve_node_type(expr.id);
        if ty::type_is_error(ty_unadjusted) || ty::type_is_bot(ty_unadjusted) {
            ty_unadjusted
        } else {
            let tcx = self.fcx.tcx();
            ty::adjust_ty(tcx, expr.span, expr.id, ty_unadjusted,
                          self.fcx.inh.adjustments.borrow().find(&expr.id),
                          |method_call| self.resolve_method_type(method_call))
        }
    }
}

impl<'fcx> mc::Typer for Rcx<'fcx> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt {
        self.fcx.tcx()
    }

    fn node_ty(&self, id: ast::NodeId) -> mc::McResult<ty::t> {
        let t = self.resolve_node_type(id);
        if ty::type_is_error(t) {Err(())} else {Ok(t)}
    }

    fn node_method_ty(&self, method_call: MethodCall) -> Option<ty::t> {
        self.resolve_method_type(method_call)
    }

    fn adjustments<'a>(&'a self) -> &'a RefCell<NodeMap<ty::AutoAdjustment>> {
        &self.fcx.inh.adjustments
    }

    fn is_method_call(&self, id: ast::NodeId) -> bool {
        self.fcx.inh.method_map.borrow().contains_key(&MethodCall::expr(id))
    }

    fn temporary_scope(&self, id: ast::NodeId) -> Option<ast::NodeId> {
        self.tcx().region_maps.temporary_scope(id)
    }

    fn upvar_borrow(&self, id: ty::UpvarId) -> ty::UpvarBorrow {
        self.fcx.inh.upvar_borrow_map.borrow().get_copy(&id)
    }
}

pub fn regionck_expr(fcx: &FnCtxt, e: &ast::Expr) {
    let mut rcx = Rcx { fcx: fcx, repeating_scope: e.id };
    let rcx = &mut rcx;
    if fcx.err_count_since_creation() == 0 {
        // regionck assumes typeck succeeded
        rcx.visit_expr(e, ());
    }
    fcx.infcx().resolve_regions_and_report_errors();
}

pub fn regionck_fn(fcx: &FnCtxt, blk: &ast::Block) {
    let mut rcx = Rcx { fcx: fcx, repeating_scope: blk.id };
    let rcx = &mut rcx;
    if fcx.err_count_since_creation() == 0 {
        // regionck assumes typeck succeeded
        rcx.visit_block(blk, ());
    }
    fcx.infcx().resolve_regions_and_report_errors();
}

impl<'a> Visitor<()> for Rcx<'a> {
    // (..) FIXME(#3238) should use visit_pat, not visit_arm/visit_local,
    // However, right now we run into an issue whereby some free
    // regions are not properly related if they appear within the
    // types of arguments that must be inferred. This could be
    // addressed by deferring the construction of the region
    // hierarchy, and in particular the relationships between free
    // regions, until regionck, as described in #3238.

    fn visit_item(&mut self, i: &ast::Item, _: ()) { visit_item(self, i); }

    fn visit_expr(&mut self, ex: &ast::Expr, _: ()) { visit_expr(self, ex); }

    //visit_pat: visit_pat, // (..) see above

    fn visit_arm(&mut self, a: &ast::Arm, _: ()) { visit_arm(self, a); }

    fn visit_local(&mut self, l: &ast::Local, _: ()) { visit_local(self, l); }

    fn visit_block(&mut self, b: &ast::Block, _: ()) { visit_block(self, b); }
}

fn visit_item(_rcx: &mut Rcx, _item: &ast::Item) {
    // Ignore items
}

fn visit_block(rcx: &mut Rcx, b: &ast::Block) {
    visit::walk_block(rcx, b, ());
}

fn visit_arm(rcx: &mut Rcx, arm: &ast::Arm) {
    // see above
    for &p in arm.pats.iter() {
        constrain_bindings_in_pat(p, rcx);
    }

    visit::walk_arm(rcx, arm, ());
}

fn visit_local(rcx: &mut Rcx, l: &ast::Local) {
    // see above
    constrain_bindings_in_pat(l.pat, rcx);
    link_local(rcx, l);
    visit::walk_local(rcx, l, ());
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
        constrain_regions_in_type_of_node(
            rcx, id, var_region,
            infer::BindingTypeIsNotValidAtDecl(span));
    })
}

fn visit_expr(rcx: &mut Rcx, expr: &ast::Expr) {
    debug!("regionck::visit_expr(e={}, repeating_scope={:?})",
           expr.repr(rcx.fcx.tcx()), rcx.repeating_scope);

    let method_call = MethodCall::expr(expr.id);
    let has_method_map = rcx.fcx.inh.method_map.borrow().contains_key(&method_call);

    // Check any autoderefs or autorefs that appear.
    for &adjustment in rcx.fcx.inh.adjustments.borrow().find(&expr.id).iter() {
        debug!("adjustment={:?}", adjustment);
        match *adjustment {
            ty::AutoDerefRef(ty::AutoDerefRef {autoderefs, autoref: opt_autoref}) => {
                let expr_ty = rcx.resolve_node_type(expr.id);
                constrain_autoderefs(rcx, expr, autoderefs, expr_ty);
                for autoref in opt_autoref.iter() {
                    link_autoref(rcx, expr, autoderefs, autoref);

                    // Require that the resulting region encompasses
                    // the current node.
                    //
                    // FIXME(#6268) remove to support nested method calls
                    constrain_regions_in_type_of_node(
                        rcx, expr.id, ty::ReScope(expr.id),
                        infer::AutoBorrow(expr.span));
                }
            }
            ty::AutoObject(ty::RegionTraitStore(trait_region, _), _, _, _) => {
                // Determine if we are casting `expr` to a trait
                // instance.  If so, we have to be sure that the type of
                // the source obeys the trait's region bound.
                //
                // Note: there is a subtle point here concerning type
                // parameters.  It is possible that the type of `source`
                // contains type parameters, which in turn may contain
                // regions that are not visible to us (only the caller
                // knows about them).  The kind checker is ultimately
                // responsible for guaranteeing region safety in that
                // particular case.  There is an extensive comment on the
                // function check_cast_for_escaping_regions() in kind.rs
                // explaining how it goes about doing that.

                let source_ty = rcx.fcx.expr_ty(expr);
                constrain_regions_in_type(rcx, trait_region,
                                            infer::RelateObjectBound(expr.span), source_ty);
            }
            _ => {}
        }
    }

    match expr.node {
        ast::ExprCall(callee, ref args) => {
            constrain_callee(rcx, callee.id, expr, callee);
            constrain_call(rcx,
                           Some(callee.id),
                           expr,
                           None,
                           args.as_slice(),
                           false);

            visit::walk_expr(rcx, expr, ());
        }

        ast::ExprMethodCall(_, _, ref args) => {
            constrain_call(rcx, None, expr, Some(*args.get(0)),
                           args.slice_from(1), false);

            visit::walk_expr(rcx, expr, ());
        }

        ast::ExprAssign(..) => {
            visit::walk_expr(rcx, expr, ());
        }

        ast::ExprAssignOp(_, lhs, rhs) => {
            if has_method_map {
                constrain_call(rcx, None, expr, Some(lhs), [rhs], true);
            }

            visit::walk_expr(rcx, expr, ());
        }

        ast::ExprIndex(lhs, rhs) |
        ast::ExprBinary(_, lhs, rhs) if has_method_map => {
            // As `expr_method_call`, but the call is via an
            // overloaded op.  Note that we (sadly) currently use an
            // implicit "by ref" sort of passing style here.  This
            // should be converted to an adjustment!
            constrain_call(rcx, None, expr, Some(lhs), [rhs], true);

            visit::walk_expr(rcx, expr, ());
        }

        ast::ExprUnary(_, lhs) if has_method_map => {
            // As above.
            constrain_call(rcx, None, expr, Some(lhs), [], true);

            visit::walk_expr(rcx, expr, ());
        }

        ast::ExprUnary(ast::UnDeref, base) => {
            // For *a, the lifetime of a must enclose the deref
            let method_call = MethodCall::expr(expr.id);
            let base_ty = match rcx.fcx.inh.method_map.borrow().find(&method_call) {
                Some(method) => {
                    constrain_call(rcx, None, expr, Some(base), [], true);
                    ty::ty_fn_ret(method.ty)
                }
                None => rcx.resolve_node_type(base.id)
            };
            match ty::get(base_ty).sty {
                ty::ty_rptr(r_ptr, _) => {
                    mk_subregion_due_to_dereference(rcx, expr.span,
                                                    ty::ReScope(expr.id), r_ptr);
                }
                _ => {}
            }

            visit::walk_expr(rcx, expr, ());
        }

        ast::ExprIndex(vec_expr, _) => {
            // For a[b], the lifetime of a must enclose the deref
            let vec_type = rcx.resolve_expr_type_adjusted(vec_expr);
            constrain_index(rcx, expr, vec_type);

            visit::walk_expr(rcx, expr, ());
        }

        ast::ExprCast(source, _) => {
            // Determine if we are casting `source` to a trait
            // instance.  If so, we have to be sure that the type of
            // the source obeys the trait's region bound.
            //
            // Note: there is a subtle point here concerning type
            // parameters.  It is possible that the type of `source`
            // contains type parameters, which in turn may contain
            // regions that are not visible to us (only the caller
            // knows about them).  The kind checker is ultimately
            // responsible for guaranteeing region safety in that
            // particular case.  There is an extensive comment on the
            // function check_cast_for_escaping_regions() in kind.rs
            // explaining how it goes about doing that.
            let target_ty = rcx.resolve_node_type(expr.id);
            match ty::get(target_ty).sty {
                ty::ty_trait(box ty::TyTrait {
                    store: ty::RegionTraitStore(trait_region, _), ..
                }) => {
                    let source_ty = rcx.resolve_expr_type_adjusted(source);
                    constrain_regions_in_type(
                        rcx,
                        trait_region,
                        infer::RelateObjectBound(expr.span),
                        source_ty);
                }
                _ => ()
            }

            visit::walk_expr(rcx, expr, ());
        }

        ast::ExprAddrOf(m, base) => {
            link_addr_of(rcx, expr, m, base);

            // Require that when you write a `&expr` expression, the
            // resulting pointer has a lifetime that encompasses the
            // `&expr` expression itself. Note that we constraining
            // the type of the node expr.id here *before applying
            // adjustments*.
            //
            // FIXME(#6268) nested method calls requires that this rule change
            let ty0 = rcx.resolve_node_type(expr.id);
            constrain_regions_in_type(rcx, ty::ReScope(expr.id),
                                      infer::AddrOf(expr.span), ty0);
            visit::walk_expr(rcx, expr, ());
        }

        ast::ExprMatch(discr, ref arms) => {
            link_match(rcx, discr, arms.as_slice());

            visit::walk_expr(rcx, expr, ());
        }

        ast::ExprFnBlock(_, ref body) | ast::ExprProc(_, ref body) => {
            check_expr_fn_block(rcx, expr, &**body);
        }

        ast::ExprLoop(body, _) => {
            let repeating_scope = rcx.set_repeating_scope(body.id);
            visit::walk_expr(rcx, expr, ());
            rcx.set_repeating_scope(repeating_scope);
        }

        ast::ExprWhile(cond, body) => {
            let repeating_scope = rcx.set_repeating_scope(cond.id);
            rcx.visit_expr(cond, ());

            rcx.set_repeating_scope(body.id);
            rcx.visit_block(body, ());

            rcx.set_repeating_scope(repeating_scope);
        }

        _ => {
            visit::walk_expr(rcx, expr, ());
        }
    }
}

fn check_expr_fn_block(rcx: &mut Rcx,
                       expr: &ast::Expr,
                       body: &ast::Block) {
    let tcx = rcx.fcx.tcx();
    let function_type = rcx.resolve_node_type(expr.id);
    match ty::get(function_type).sty {
        ty::ty_closure(box ty::ClosureTy {
                store: ty::RegionTraitStore(region, _), ..}) => {
            freevars::with_freevars(tcx, expr.id, |freevars| {
                if freevars.is_empty() {
                    // No free variables means that the environment
                    // will be NULL at runtime and hence the closure
                    // has static lifetime.
                } else {
                    // Closure cannot outlive the appropriate temporary scope.
                    let s = rcx.repeating_scope;
                    rcx.fcx.mk_subr(true, infer::InfStackClosure(expr.span),
                                    region, ty::ReScope(s));
                }
            });
        }
        _ => ()
    }

    let repeating_scope = rcx.set_repeating_scope(body.id);
    visit::walk_expr(rcx, expr, ());
    rcx.set_repeating_scope(repeating_scope);
}

fn constrain_callee(rcx: &mut Rcx,
                    callee_id: ast::NodeId,
                    call_expr: &ast::Expr,
                    callee_expr: &ast::Expr) {
    let call_region = ty::ReScope(call_expr.id);

    let callee_ty = rcx.resolve_node_type(callee_id);
    match ty::get(callee_ty).sty {
        ty::ty_bare_fn(..) => { }
        ty::ty_closure(ref closure_ty) => {
            let region = match closure_ty.store {
                ty::RegionTraitStore(r, _) => {
                    // While we're here, link the closure's region with a unique
                    // immutable borrow (gathered later in borrowck)
                    let mc = mc::MemCategorizationContext::new(rcx);
                    let expr_cmt = ignore_err!(mc.cat_expr(callee_expr));
                    link_region(rcx, callee_expr.span, call_region,
                                ty::UniqueImmBorrow, expr_cmt);
                    r
                }
                ty::UniqTraitStore => ty::ReStatic
            };
            rcx.fcx.mk_subr(true, infer::InvokeClosure(callee_expr.span),
                            call_region, region);
        }
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

fn constrain_call(rcx: &mut Rcx,
                  // might be expr_call, expr_method_call, or an overloaded
                  // operator
                  fn_expr_id: Option<ast::NodeId>,
                  call_expr: &ast::Expr,
                  receiver: Option<@ast::Expr>,
                  arg_exprs: &[@ast::Expr],
                  implicitly_ref_args: bool) {
    //! Invoked on every call site (i.e., normal calls, method calls,
    //! and overloaded operators). Constrains the regions which appear
    //! in the type of the function. Also constrains the regions that
    //! appear in the arguments appropriately.

    let tcx = rcx.fcx.tcx();
    debug!("constrain_call(call_expr={}, \
            receiver={}, \
            arg_exprs={}, \
            implicitly_ref_args={:?})",
            call_expr.repr(tcx),
            receiver.repr(tcx),
            arg_exprs.repr(tcx),
            implicitly_ref_args);
    let callee_ty = match fn_expr_id {
        Some(id) => rcx.resolve_node_type(id),
        None => rcx.resolve_method_type(MethodCall::expr(call_expr.id))
                   .expect("call should have been to a method")
    };
    if ty::type_is_error(callee_ty) {
        // Bail, as function type is unknown
        return;
    }
    let fn_sig = ty::ty_fn_sig(callee_ty);

    // `callee_region` is the scope representing the time in which the
    // call occurs.
    //
    // FIXME(#6268) to support nested method calls, should be callee_id
    let callee_scope = call_expr.id;
    let callee_region = ty::ReScope(callee_scope);

    for &arg_expr in arg_exprs.iter() {
        debug!("Argument");

        // ensure that any regions appearing in the argument type are
        // valid for at least the lifetime of the function:
        constrain_regions_in_type_of_node(
            rcx, arg_expr.id, callee_region,
            infer::CallArg(arg_expr.span));

        // unfortunately, there are two means of taking implicit
        // references, and we need to propagate constraints as a
        // result. modes are going away and the "DerefArgs" code
        // should be ported to use adjustments
        if implicitly_ref_args {
            link_by_ref(rcx, arg_expr, callee_scope);
        }
    }

    // as loop above, but for receiver
    for &r in receiver.iter() {
        debug!("Receiver");
        constrain_regions_in_type_of_node(
            rcx, r.id, callee_region, infer::CallRcvr(r.span));
        if implicitly_ref_args {
            link_by_ref(rcx, r, callee_scope);
        }
    }

    // constrain regions that may appear in the return type to be
    // valid for the function call:
    constrain_regions_in_type(
        rcx, callee_region, infer::CallReturn(call_expr.span),
        fn_sig.output);
}

fn constrain_autoderefs(rcx: &mut Rcx,
                        deref_expr: &ast::Expr,
                        derefs: uint,
                        mut derefd_ty: ty::t) {
    /*!
     * Invoked on any auto-dereference that occurs.  Checks that if
     * this is a region pointer being dereferenced, the lifetime of
     * the pointer includes the deref expr.
     */
    let r_deref_expr = ty::ReScope(deref_expr.id);
    for i in range(0u, derefs) {
        debug!("constrain_autoderefs(deref_expr=?, derefd_ty={}, derefs={:?}/{:?}",
               rcx.fcx.infcx().ty_to_str(derefd_ty),
               i, derefs);

        let method_call = MethodCall::autoderef(deref_expr.id, i as u32);
        derefd_ty = match rcx.fcx.inh.method_map.borrow().find(&method_call) {
            Some(method) => {
                // Treat overloaded autoderefs as if an AutoRef adjustment
                // was applied on the base type, as that is always the case.
                let fn_sig = ty::ty_fn_sig(method.ty);
                let self_ty = *fn_sig.inputs.get(0);
                let (m, r) = match ty::get(self_ty).sty {
                    ty::ty_rptr(r, ref m) => (m.mutbl, r),
                    _ => rcx.tcx().sess.span_bug(deref_expr.span,
                            format!("bad overloaded deref type {}",
                                    method.ty.repr(rcx.tcx())).as_slice())
                };
                {
                    let mc = mc::MemCategorizationContext::new(rcx);
                    let self_cmt = ignore_err!(mc.cat_expr_autoderefd(deref_expr, i));
                    link_region(rcx, deref_expr.span, r,
                                ty::BorrowKind::from_mutbl(m), self_cmt);
                }

                // Specialized version of constrain_call.
                constrain_regions_in_type(rcx, r_deref_expr,
                                          infer::CallRcvr(deref_expr.span),
                                          self_ty);
                constrain_regions_in_type(rcx, r_deref_expr,
                                          infer::CallReturn(deref_expr.span),
                                          fn_sig.output);
                fn_sig.output
            }
            None => derefd_ty
        };

        match ty::get(derefd_ty).sty {
            ty::ty_rptr(r_ptr, _) => {
                mk_subregion_due_to_dereference(rcx, deref_expr.span,
                                                r_deref_expr, r_ptr);
            }
            _ => {}
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
    rcx.fcx.mk_subr(true, infer::DerefPointer(deref_span),
                    minimum_lifetime, maximum_lifetime)
}


fn constrain_index(rcx: &mut Rcx,
                   index_expr: &ast::Expr,
                   indexed_ty: ty::t)
{
    /*!
     * Invoked on any index expression that occurs.  Checks that if
     * this is a slice being indexed, the lifetime of the pointer
     * includes the deref expr.
     */

    debug!("constrain_index(index_expr=?, indexed_ty={}",
           rcx.fcx.infcx().ty_to_str(indexed_ty));

    let r_index_expr = ty::ReScope(index_expr.id);
    match ty::get(indexed_ty).sty {
        ty::ty_rptr(r_ptr, mt) => match ty::get(mt.ty).sty {
            ty::ty_vec(_, None) | ty::ty_str => {
                rcx.fcx.mk_subr(true, infer::IndexSlice(index_expr.span),
                                r_index_expr, r_ptr);
            }
            _ => {}
        },

        _ => {}
    }
}

fn constrain_regions_in_type_of_node(
    rcx: &mut Rcx,
    id: ast::NodeId,
    minimum_lifetime: ty::Region,
    origin: infer::SubregionOrigin) {
    //! Guarantees that any lifetimes which appear in the type of
    //! the node `id` (after applying adjustments) are valid for at
    //! least `minimum_lifetime`

    let tcx = rcx.fcx.tcx();

    // Try to resolve the type.  If we encounter an error, then typeck
    // is going to fail anyway, so just stop here and let typeck
    // report errors later on in the writeback phase.
    let ty0 = rcx.resolve_node_type(id);
    let ty = ty::adjust_ty(tcx, origin.span(), id, ty0,
                           rcx.fcx.inh.adjustments.borrow().find(&id),
                           |method_call| rcx.resolve_method_type(method_call));
    debug!("constrain_regions_in_type_of_node(\
            ty={}, ty0={}, id={}, minimum_lifetime={:?})",
           ty_to_str(tcx, ty), ty_to_str(tcx, ty0),
           id, minimum_lifetime);
    constrain_regions_in_type(rcx, minimum_lifetime, origin, ty);
}

fn constrain_regions_in_type(
    rcx: &mut Rcx,
    minimum_lifetime: ty::Region,
    origin: infer::SubregionOrigin,
    ty: ty::t) {
    /*!
     * Requires that any regions which appear in `ty` must be
     * superregions of `minimum_lifetime`.  Also enforces the constraint
     * that given a pointer type `&'r T`, T must not contain regions
     * that outlive 'r, as well as analogous constraints for other
     * lifetime'd types.
     *
     * This check prevents regions from being used outside of the block in
     * which they are valid.  Recall that regions represent blocks of
     * code or expressions: this requirement basically says "any place
     * that uses or may use a region R must be within the block of
     * code that R corresponds to."
     */

    let tcx = rcx.fcx.ccx.tcx;

    debug!("constrain_regions_in_type(minimum_lifetime={}, ty={})",
           region_to_str(tcx, "", false, minimum_lifetime),
           ty_to_str(tcx, ty));

    relate_nested_regions(tcx, Some(minimum_lifetime), ty, |r_sub, r_sup| {
        debug!("relate_nested_regions(r_sub={}, r_sup={})",
                r_sub.repr(tcx),
                r_sup.repr(tcx));

        if r_sup.is_bound() || r_sub.is_bound() {
            // a bound region is one which appears inside an fn type.
            // (e.g., the `&` in `fn(&T)`).  Such regions need not be
            // constrained by `minimum_lifetime` as they are placeholders
            // for regions that are as-yet-unknown.
        } else if r_sub == minimum_lifetime {
            rcx.fcx.mk_subr(
                true, origin.clone(),
                r_sub, r_sup);
        } else {
            rcx.fcx.mk_subr(
                true, infer::ReferenceOutlivesReferent(ty, origin.span()),
                r_sub, r_sup);
        }
    });
}

fn link_addr_of(rcx: &mut Rcx, expr: &ast::Expr,
               mutability: ast::Mutability, base: &ast::Expr) {
    /*!
     * Computes the guarantor for an expression `&base` and then
     * ensures that the lifetime of the resulting pointer is linked
     * to the lifetime of its guarantor (if any).
     */

    debug!("link_addr_of(base=?)");

    let cmt = {
        let mc = mc::MemCategorizationContext::new(rcx);
        ignore_err!(mc.cat_expr(base))
    };
    link_region_from_node_type(rcx, expr.span, expr.id, mutability, cmt);
}

fn link_local(rcx: &Rcx, local: &ast::Local) {
    /*!
     * Computes the guarantors for any ref bindings in a `let` and
     * then ensures that the lifetime of the resulting pointer is
     * linked to the lifetime of the initialization expression.
     */

    debug!("regionck::for_local()");
    let init_expr = match local.init {
        None => { return; }
        Some(expr) => expr,
    };
    let mc = mc::MemCategorizationContext::new(rcx);
    let discr_cmt = ignore_err!(mc.cat_expr(init_expr));
    link_pattern(rcx, mc, discr_cmt, local.pat);
}

fn link_match(rcx: &Rcx, discr: &ast::Expr, arms: &[ast::Arm]) {
    /*!
     * Computes the guarantors for any ref bindings in a match and
     * then ensures that the lifetime of the resulting pointer is
     * linked to the lifetime of its guarantor (if any).
     */

    debug!("regionck::for_match()");
    let mc = mc::MemCategorizationContext::new(rcx);
    let discr_cmt = ignore_err!(mc.cat_expr(discr));
    debug!("discr_cmt={}", discr_cmt.repr(rcx.tcx()));
    for arm in arms.iter() {
        for &root_pat in arm.pats.iter() {
            link_pattern(rcx, mc, discr_cmt.clone(), root_pat);
        }
    }
}

fn link_pattern(rcx: &Rcx,
                mc: mc::MemCategorizationContext<Rcx>,
                discr_cmt: mc::cmt,
                root_pat: &ast::Pat) {
    /*!
     * Link lifetimes of any ref bindings in `root_pat` to
     * the pointers found in the discriminant, if needed.
     */

    let _ = mc.cat_pattern(discr_cmt, root_pat, |mc, sub_cmt, sub_pat| {
            match sub_pat.node {
                // `ref x` pattern
                ast::PatIdent(ast::BindByRef(mutbl), _, _) => {
                    link_region_from_node_type(
                        rcx, sub_pat.span, sub_pat.id,
                        mutbl, sub_cmt);
                }

                // `[_, ..slice, _]` pattern
                ast::PatVec(_, Some(slice_pat), _) => {
                    match mc.cat_slice_pattern(sub_cmt, slice_pat) {
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

fn link_autoref(rcx: &Rcx,
                expr: &ast::Expr,
                autoderefs: uint,
                autoref: &ty::AutoRef) {
    /*!
     * Link lifetime of borrowed pointer resulting from autoref
     * to lifetimes in the value being autoref'd.
     */

    debug!("link_autoref(autoref={:?})", autoref);
    let mc = mc::MemCategorizationContext::new(rcx);
    let expr_cmt = ignore_err!(mc.cat_expr_autoderefd(expr, autoderefs));
    debug!("expr_cmt={}", expr_cmt.repr(rcx.tcx()));

    match *autoref {
        ty::AutoPtr(r, m) => {
            link_region(rcx, expr.span, r,
                        ty::BorrowKind::from_mutbl(m), expr_cmt);
        }

        ty::AutoBorrowVec(r, m) | ty::AutoBorrowVecRef(r, m) => {
            let cmt_index = mc.cat_index(expr, expr_cmt, autoderefs+1);
            link_region(rcx, expr.span, r,
                        ty::BorrowKind::from_mutbl(m), cmt_index);
        }

        ty::AutoBorrowObj(r, m) => {
            let cmt_deref = mc.cat_deref_obj(expr, expr_cmt);
            link_region(rcx, expr.span, r,
                        ty::BorrowKind::from_mutbl(m), cmt_deref);
        }

        ty::AutoUnsafe(_) => {}
    }
}

fn link_by_ref(rcx: &Rcx,
               expr: &ast::Expr,
               callee_scope: ast::NodeId) {
    /*!
     * Computes the guarantor for cases where the `expr` is
     * being passed by implicit reference and must outlive
     * `callee_scope`.
     */

    let tcx = rcx.tcx();
    debug!("link_by_ref(expr={}, callee_scope={})",
           expr.repr(tcx), callee_scope);
    let mc = mc::MemCategorizationContext::new(rcx);
    let expr_cmt = ignore_err!(mc.cat_expr(expr));
    let region_min = ty::ReScope(callee_scope);
    link_region(rcx, expr.span, region_min, ty::ImmBorrow, expr_cmt);
}

fn link_region_from_node_type(rcx: &Rcx,
                              span: Span,
                              id: ast::NodeId,
                              mutbl: ast::Mutability,
                              cmt_borrowed: mc::cmt) {
    /*!
     * Like `link_region()`, except that the region is
     * extracted from the type of `id`, which must be some
     * reference (`&T`, `&str`, etc).
     */

    let rptr_ty = rcx.resolve_node_type(id);
    if !ty::type_is_bot(rptr_ty) && !ty::type_is_error(rptr_ty) {
        let tcx = rcx.fcx.ccx.tcx;
        debug!("rptr_ty={}", ty_to_str(tcx, rptr_ty));
        let r = ty::ty_region(tcx, span, rptr_ty);
        link_region(rcx, span, r, ty::BorrowKind::from_mutbl(mutbl),
                    cmt_borrowed);
    }
}

fn link_region(rcx: &Rcx,
               span: Span,
               region_min: ty::Region,
               kind: ty::BorrowKind,
               cmt_borrowed: mc::cmt) {
    /*!
     * Informs the inference engine that a borrow of `cmt`
     * must have the borrow kind `kind` and lifetime `region_min`.
     * If `cmt` is a deref of a region pointer with
     * lifetime `r_borrowed`, this will add the constraint that
     * `region_min <= r_borrowed`.
     */

    // Iterate through all the things that must be live at least
    // for the lifetime `region_min` for the borrow to be valid:
    let mut cmt_borrowed = cmt_borrowed;
    loop {
        debug!("link_region(region_min={}, kind={}, cmt_borrowed={})",
               region_min.repr(rcx.tcx()),
               kind.repr(rcx.tcx()),
               cmt_borrowed.repr(rcx.tcx()));
        match cmt_borrowed.cat.clone() {
            mc::cat_deref(_, _, mc::BorrowedPtr(_, r_borrowed)) => {
                let cause = infer::Reborrow(span);

                debug!("link_region: {} <= {}",
                       region_min.repr(rcx.tcx()),
                       r_borrowed.repr(rcx.tcx()));
                rcx.fcx.mk_subr(true, cause, region_min, r_borrowed);

                // Borrowing an `&mut` pointee for `region_min` is
                // only valid if the pointer resides in a unique
                // location which is itself valid for
                // `region_min`.  We don't care about the unique
                // part, but we may need to influence the
                // inference to ensure that the location remains
                // valid.
                //
                // FIXME(#8624) fixing borrowck will require this
                // if m == ast::m_mutbl {
                //    cmt_borrowed = cmt_base;
                // } else {
                //    return;
                // }
                return;
            }
            mc::cat_discr(cmt_base, _) |
            mc::cat_downcast(cmt_base) |
            mc::cat_deref(cmt_base, _, mc::GcPtr(..)) |
            mc::cat_deref(cmt_base, _, mc::OwnedPtr) |
            mc::cat_interior(cmt_base, _) => {
                // Interior or owned data requires its base to be valid
                cmt_borrowed = cmt_base;
            }
            mc::cat_deref(_, _, mc::UnsafePtr(..)) |
            mc::cat_static_item |
            mc::cat_copied_upvar(..) |
            mc::cat_local(..) |
            mc::cat_arg(..) |
            mc::cat_rvalue(..) => {
                // These are all "base cases" with independent lifetimes
                // that are not subject to inference
                return;
            }
        }
    }
}

