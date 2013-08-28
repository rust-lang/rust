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
the types.  Its purpose is to embed some final region constraints.
The reason that this is not done earlier is that sometimes we don't
know whether a given type will be a region pointer or not until this
phase.

In particular, we ensure that, if the type of an expression or
variable is `&'r T`, then the expression or variable must occur within
the region scope `r`.  Note that in some cases `r` may still be a
region variable, so this gives us a chance to influence the value for
`r` that we infer to ensure we choose a value large enough to enclose
all uses.  There is a lengthy comment in visit_node() that explains
this point a bit better.

*/


use middle::freevars::get_freevars;
use middle::ty::{re_scope};
use middle::ty;
use middle::typeck::check::FnCtxt;
use middle::typeck::check::regionmanip::relate_nested_regions;
use middle::typeck::infer::resolve_and_force_all_but_regions;
use middle::typeck::infer::resolve_type;
use middle::typeck::infer;
use util::ppaux::{ty_to_str, region_to_str};
use middle::pat_util;

use syntax::ast::{ManagedSigil, OwnedSigil, BorrowedSigil};
use syntax::ast::{DefArg, DefBinding, DefLocal, DefSelf, DefUpvar};
use syntax::ast;
use syntax::codemap::Span;
use syntax::visit;
use syntax::visit::Visitor;

pub struct Rcx {
    fcx: @mut FnCtxt,
    errors_reported: uint,

    // id of innermost fn or loop
    repeating_scope: ast::NodeId,
}

fn encl_region_of_def(fcx: @mut FnCtxt, def: ast::Def) -> ty::Region {
    let tcx = fcx.tcx();
    match def {
        DefLocal(node_id, _) | DefArg(node_id, _) |
        DefSelf(node_id) | DefBinding(node_id, _) => {
            tcx.region_maps.encl_region(node_id)
        }
        DefUpvar(_, subdef, closure_id, body_id) => {
            match ty::ty_closure_sigil(fcx.node_ty(closure_id)) {
                BorrowedSigil => encl_region_of_def(fcx, *subdef),
                ManagedSigil | OwnedSigil => re_scope(body_id)
            }
        }
        _ => {
            tcx.sess.bug(fmt!("unexpected def in encl_region_of_def: %?",
                              def))
        }
    }
}

impl Rcx {
    pub fn tcx(&self) -> ty::ctxt {
        self.fcx.ccx.tcx
    }

    pub fn set_repeating_scope(&mut self, scope: ast::NodeId) -> ast::NodeId {
        let old_scope = self.repeating_scope;
        self.repeating_scope = scope;
        old_scope
    }

    pub fn resolve_type(&mut self, unresolved_ty: ty::t) -> ty::t {
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
    pub fn resolve_node_type(@mut self, id: ast::NodeId) -> ty::t {
        self.resolve_type(self.fcx.node_ty(id))
    }

    /// Try to resolve the type for the given node.
    pub fn resolve_expr_type_adjusted(@mut self, expr: @ast::Expr) -> ty::t {
        let ty_unadjusted = self.resolve_node_type(expr.id);
        if ty::type_is_error(ty_unadjusted) || ty::type_is_bot(ty_unadjusted) {
            ty_unadjusted
        } else {
            let tcx = self.fcx.tcx();
            let adjustments = self.fcx.inh.adjustments;
            ty::adjust_ty(tcx, expr.span, ty_unadjusted,
                          adjustments.find_copy(&expr.id))
        }
    }
}

pub fn regionck_expr(fcx: @mut FnCtxt, e: @ast::Expr) {
    let rcx = @mut Rcx { fcx: fcx, errors_reported: 0,
                         repeating_scope: e.id };
    if fcx.err_count_since_creation() == 0 {
        // regionck assumes typeck succeeded
        let mut v = regionck_visitor();
        v.visit_expr(e, rcx);
    }
    fcx.infcx().resolve_regions();
}

pub fn regionck_fn(fcx: @mut FnCtxt, blk: &ast::Block) {
    let rcx = @mut Rcx { fcx: fcx, errors_reported: 0,
                         repeating_scope: blk.id };
    if fcx.err_count_since_creation() == 0 {
        // regionck assumes typeck succeeded
        let mut v = regionck_visitor();
        v.visit_block(blk, rcx);
    }
    fcx.infcx().resolve_regions();
}

struct RegionckVisitor;

impl Visitor<@mut Rcx> for RegionckVisitor {
    // (*) FIXME(#3238) should use visit_pat, not visit_arm/visit_local,
    // However, right now we run into an issue whereby some free
    // regions are not properly related if they appear within the
    // types of arguments that must be inferred. This could be
    // addressed by deferring the construction of the region
    // hierarchy, and in particular the relationships between free
    // regions, until regionck, as described in #3238.

    fn visit_item(&mut self, i:@ast::item, e:@mut Rcx) { visit_item(self, i, e); }

    fn visit_expr(&mut self, ex:@ast::Expr, e:@mut Rcx) { visit_expr(self, ex, e); }

        //visit_pat: visit_pat, // (*) see above

    fn visit_arm(&mut self, a:&ast::Arm, e:@mut Rcx) { visit_arm(self, a, e); }

    fn visit_local(&mut self, l:@ast::Local, e:@mut Rcx) { visit_local(self, l, e); }

    fn visit_block(&mut self, b:&ast::Block, e:@mut Rcx) { visit_block(self, b, e); }
}

fn regionck_visitor() -> RegionckVisitor {
    RegionckVisitor
}

fn visit_item(_v: &mut RegionckVisitor, _item: @ast::item, _rcx: @mut Rcx) {
    // Ignore items
}

fn visit_block(v: &mut RegionckVisitor, b: &ast::Block, rcx: @mut Rcx) {
    rcx.fcx.tcx().region_maps.record_cleanup_scope(b.id);
    visit::walk_block(v, b, rcx);
}

fn visit_arm(v: &mut RegionckVisitor, arm: &ast::Arm, rcx: @mut Rcx) {
    // see above
    for &p in arm.pats.iter() {
        constrain_bindings_in_pat(p, rcx);
    }

    visit::walk_arm(v, arm, rcx);
}

fn visit_local(v: &mut RegionckVisitor, l: @ast::Local, rcx: @mut Rcx) {
    // see above
    constrain_bindings_in_pat(l.pat, rcx);
    visit::walk_local(v, l, rcx);
}

fn constrain_bindings_in_pat(pat: @ast::Pat, rcx: @mut Rcx) {
    let tcx = rcx.fcx.tcx();
    debug!("regionck::visit_pat(pat=%s)", pat.repr(tcx));
    do pat_util::pat_bindings(tcx.def_map, pat) |_, id, span, _| {
        // If we have a variable that contains region'd data, that
        // data will be accessible from anywhere that the variable is
        // accessed. We must be wary of loops like this:
        //
        //     // from src/test/compile-fail/borrowck-lend-flow.rs
        //     let mut v = ~3, w = ~4;
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

        let encl_region = tcx.region_maps.encl_region(id);
        constrain_regions_in_type_of_node(
            rcx, id, encl_region,
            infer::BindingTypeIsNotValidAtDecl(span));
    }
}

fn visit_expr(v: &mut RegionckVisitor, expr: @ast::Expr, rcx: @mut Rcx) {
    debug!("regionck::visit_expr(e=%s, repeating_scope=%?)",
           expr.repr(rcx.fcx.tcx()), rcx.repeating_scope);

    let has_method_map = rcx.fcx.inh.method_map.contains_key(&expr.id);

    // Record cleanup scopes, which are used by borrowck to decide the
    // maximum lifetime of a temporary rvalue.  These were derived by
    // examining where trans creates block scopes, not because this
    // reflects some principled decision around temporary lifetimes.
    // Ordinarily this would seem like something that should be setup
    // in region, but we need to know which uses of operators are
    // overloaded.  See #3511.
    let tcx = rcx.fcx.tcx();
    match expr.node {
        // You'd think that x += y where `+=` is overloaded would be a
        // cleanup scope. You'd be... kind of right. In fact the
        // handling of `+=` and friends in trans for overloaded
        // operators is a hopeless mess and I can't figure out how to
        // represent it. - ndm
        //
        // ast::expr_assign_op(*) |

        ast::ExprIndex(*) |
        ast::ExprBinary(*) |
        ast::ExprUnary(*) if has_method_map => {
            tcx.region_maps.record_cleanup_scope(expr.id);
        }
        ast::ExprBinary(_, ast::BiAnd, lhs, rhs) |
        ast::ExprBinary(_, ast::BiOr, lhs, rhs) => {
            tcx.region_maps.record_cleanup_scope(lhs.id);
            tcx.region_maps.record_cleanup_scope(rhs.id);
        }
        ast::ExprCall(*) |
        ast::ExprMethodCall(*) => {
            tcx.region_maps.record_cleanup_scope(expr.id);
        }
        ast::ExprMatch(_, ref arms) => {
            tcx.region_maps.record_cleanup_scope(expr.id);
            for arm in arms.iter() {
                for guard in arm.guard.iter() {
                    tcx.region_maps.record_cleanup_scope(guard.id);
                }
            }
        }
        ast::ExprLoop(ref body, _) => {
            tcx.region_maps.record_cleanup_scope(body.id);
        }
        ast::ExprWhile(cond, ref body) => {
            tcx.region_maps.record_cleanup_scope(cond.id);
            tcx.region_maps.record_cleanup_scope(body.id);
        }
        _ => {}
    }

    // Check any autoderefs or autorefs that appear.
    {
        let r = rcx.fcx.inh.adjustments.find(&expr.id);
        for &adjustment in r.iter() {
            debug!("adjustment=%?", adjustment);
            match *adjustment {
                @ty::AutoDerefRef(
                    ty::AutoDerefRef {autoderefs: autoderefs, autoref: opt_autoref}) =>
                {
                    let expr_ty = rcx.resolve_node_type(expr.id);
                    constrain_derefs(rcx, expr, autoderefs, expr_ty);
                    for autoref in opt_autoref.iter() {
                        guarantor::for_autoref(rcx, expr, autoderefs, autoref);

                        // Require that the resulting region encompasses
                        // the current node.
                        //
                        // FIXME(#6268) remove to support nested method calls
                        constrain_regions_in_type_of_node(
                            rcx, expr.id, ty::re_scope(expr.id),
                            infer::AutoBorrow(expr.span));
                    }
                }
                _ => {}
            }
        }
    }

    match expr.node {
        ast::ExprCall(callee, ref args, _) => {
            constrain_callee(rcx, callee.id, expr, callee);
            constrain_call(rcx, callee.id, expr, None, *args, false);

            visit::walk_expr(v, expr, rcx);
        }

        ast::ExprMethodCall(callee_id, arg0, _, _, ref args, _) => {
            constrain_call(rcx, callee_id, expr, Some(arg0), *args, false);

            visit::walk_expr(v,expr, rcx);
        }

        ast::ExprIndex(callee_id, lhs, rhs) |
        ast::ExprAssignOp(callee_id, _, lhs, rhs) |
        ast::ExprBinary(callee_id, _, lhs, rhs) if has_method_map => {
            // As `expr_method_call`, but the call is via an
            // overloaded op.  Note that we (sadly) currently use an
            // implicit "by ref" sort of passing style here.  This
            // should be converted to an adjustment!
            constrain_call(rcx, callee_id, expr, Some(lhs), [rhs], true);

            visit::walk_expr(v, expr, rcx);
        }

        ast::ExprUnary(callee_id, _, lhs) if has_method_map => {
            // As above.
            constrain_call(rcx, callee_id, expr, Some(lhs), [], true);

            visit::walk_expr(v, expr, rcx);
        }

        ast::ExprUnary(_, ast::UnDeref, base) => {
            // For *a, the lifetime of a must enclose the deref
            let base_ty = rcx.resolve_node_type(base.id);
            constrain_derefs(rcx, expr, 1, base_ty);

            visit::walk_expr(v, expr, rcx);
        }

        ast::ExprIndex(_, vec_expr, _) => {
            // For a[b], the lifetime of a must enclose the deref
            let vec_type = rcx.resolve_expr_type_adjusted(vec_expr);
            constrain_index(rcx, expr, vec_type);

            visit::walk_expr(v, expr, rcx);
        }

        ast::ExprCast(source, _) => {
            // Determine if we are casting `source` to an trait
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
                ty::ty_trait(_, _, ty::RegionTraitStore(trait_region), _, _) => {
                    let source_ty = rcx.resolve_expr_type_adjusted(source);
                    constrain_regions_in_type(
                        rcx,
                        trait_region,
                        infer::RelateObjectBound(expr.span),
                        source_ty);
                }
                _ => ()
            }

            visit::walk_expr(v, expr, rcx);
        }

        ast::ExprAddrOf(_, base) => {
            guarantor::for_addr_of(rcx, expr, base);

            // Require that when you write a `&expr` expression, the
            // resulting pointer has a lifetime that encompasses the
            // `&expr` expression itself. Note that we constraining
            // the type of the node expr.id here *before applying
            // adjustments*.
            //
            // FIXME(#6268) nested method calls requires that this rule change
            let ty0 = rcx.resolve_node_type(expr.id);
            constrain_regions_in_type(rcx, ty::re_scope(expr.id),
                                      infer::AddrOf(expr.span), ty0);
            visit::walk_expr(v, expr, rcx);
        }

        ast::ExprMatch(discr, ref arms) => {
            guarantor::for_match(rcx, discr, *arms);

            visit::walk_expr(v, expr, rcx);
        }

        ast::ExprFnBlock(*) => {
            check_expr_fn_block(rcx, expr, v);
        }

        ast::ExprLoop(ref body, _) => {
            let repeating_scope = rcx.set_repeating_scope(body.id);
            visit::walk_expr(v, expr, rcx);
            rcx.set_repeating_scope(repeating_scope);
        }

        ast::ExprWhile(cond, ref body) => {
            let repeating_scope = rcx.set_repeating_scope(cond.id);
            v.visit_expr(cond, rcx);

            rcx.set_repeating_scope(body.id);
            v.visit_block(body, rcx);

            rcx.set_repeating_scope(repeating_scope);
        }

        _ => {
            visit::walk_expr(v, expr, rcx);
        }
    }
}

fn check_expr_fn_block(rcx: @mut Rcx,
                       expr: @ast::Expr,
                       v: &mut RegionckVisitor) {
    let tcx = rcx.fcx.tcx();
    match expr.node {
        ast::ExprFnBlock(_, ref body) => {
            let function_type = rcx.resolve_node_type(expr.id);
            match ty::get(function_type).sty {
                ty::ty_closure(
                    ty::ClosureTy {
                        sigil: ast::BorrowedSigil, region: region, _}) => {
                    if get_freevars(tcx, expr.id).is_empty() {
                        // No free variables means that the environment
                        // will be NULL at runtime and hence the closure
                        // has static lifetime.
                    } else {
                        // Otherwise, the closure must not outlive the
                        // variables it closes over, nor can it
                        // outlive the innermost repeating scope
                        // (since otherwise that would require
                        // infinite stack).
                        constrain_free_variables(rcx, region, expr);
                        let repeating_scope = ty::re_scope(rcx.repeating_scope);
                        rcx.fcx.mk_subr(true, infer::InfStackClosure(expr.span),
                                        region, repeating_scope);
                    }
                }
                _ => ()
            }

            let repeating_scope = rcx.set_repeating_scope(body.id);
            visit::walk_expr(v, expr, rcx);
            rcx.set_repeating_scope(repeating_scope);
        }

        _ => {
            tcx.sess.span_bug(
                expr.span,
                "Expected expr_fn_block");
        }
    }
}

fn constrain_callee(rcx: @mut Rcx,
                    callee_id: ast::NodeId,
                    call_expr: @ast::Expr,
                    callee_expr: @ast::Expr)
{
    let call_region = ty::re_scope(call_expr.id);

    let callee_ty = rcx.resolve_node_type(callee_id);
    match ty::get(callee_ty).sty {
        ty::ty_bare_fn(*) => { }
        ty::ty_closure(ref closure_ty) => {
            rcx.fcx.mk_subr(true, infer::InvokeClosure(callee_expr.span),
                            call_region, closure_ty.region);
        }
        _ => {
            // this should not happen, but it does if the program is
            // erroneous
            //
            // tcx.sess.span_bug(
            //     callee_expr.span,
            //     fmt!("Calling non-function: %s", callee_ty.repr(tcx)));
        }
    }
}

fn constrain_call(rcx: @mut Rcx,
                  // might be expr_call, expr_method_call, or an overloaded
                  // operator
                  callee_id: ast::NodeId,
                  call_expr: @ast::Expr,
                  receiver: Option<@ast::Expr>,
                  arg_exprs: &[@ast::Expr],
                  implicitly_ref_args: bool)
{
    //! Invoked on every call site (i.e., normal calls, method calls,
    //! and overloaded operators). Constrains the regions which appear
    //! in the type of the function. Also constrains the regions that
    //! appear in the arguments appropriately.

    let tcx = rcx.fcx.tcx();
    debug!("constrain_call(call_expr=%s, implicitly_ref_args=%?)",
           call_expr.repr(tcx), implicitly_ref_args);
    let callee_ty = rcx.resolve_node_type(callee_id);
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
    let callee_region = ty::re_scope(callee_scope);

    for &arg_expr in arg_exprs.iter() {
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
            guarantor::for_by_ref(rcx, arg_expr, callee_scope);
        }
    }

    // as loop above, but for receiver
    for &r in receiver.iter() {
        constrain_regions_in_type_of_node(
            rcx, r.id, callee_region, infer::CallRcvr(r.span));
        if implicitly_ref_args {
            guarantor::for_by_ref(rcx, r, callee_scope);
        }
    }

    // constrain regions that may appear in the return type to be
    // valid for the function call:
    constrain_regions_in_type(
        rcx, callee_region, infer::CallReturn(call_expr.span),
        fn_sig.output);
}

fn constrain_derefs(rcx: @mut Rcx,
                    deref_expr: @ast::Expr,
                    derefs: uint,
                    mut derefd_ty: ty::t)
{
    /*!
     * Invoked on any dereference that occurs, whether explicitly
     * or through an auto-deref.  Checks that if this is a region
     * pointer being derefenced, the lifetime of the pointer includes
     * the deref expr.
     */
    let tcx = rcx.fcx.tcx();
    let r_deref_expr = ty::re_scope(deref_expr.id);
    for i in range(0u, derefs) {
        debug!("constrain_derefs(deref_expr=?, derefd_ty=%s, derefs=%?/%?",
               rcx.fcx.infcx().ty_to_str(derefd_ty),
               i, derefs);

        match ty::get(derefd_ty).sty {
            ty::ty_rptr(r_ptr, _) => {
                mk_subregion_due_to_derefence(rcx, deref_expr.span,
                                              r_deref_expr, r_ptr);
            }

            _ => {}
        }

        match ty::deref(tcx, derefd_ty, true) {
            Some(mt) => derefd_ty = mt.ty,
            /* if this type can't be dereferenced, then there's already an error
               in the session saying so. Just bail out for now */
            None => break
        }
    }
}

pub fn mk_subregion_due_to_derefence(rcx: @mut Rcx,
                                     deref_span: Span,
                                     minimum_lifetime: ty::Region,
                                     maximum_lifetime: ty::Region) {
    rcx.fcx.mk_subr(true, infer::DerefPointer(deref_span),
                    minimum_lifetime, maximum_lifetime)
}


fn constrain_index(rcx: @mut Rcx,
                   index_expr: @ast::Expr,
                   indexed_ty: ty::t)
{
    /*!
     * Invoked on any index expression that occurs.  Checks that if
     * this is a slice being indexed, the lifetime of the pointer
     * includes the deref expr.
     */

    debug!("constrain_index(index_expr=?, indexed_ty=%s",
           rcx.fcx.infcx().ty_to_str(indexed_ty));

    let r_index_expr = ty::re_scope(index_expr.id);
    match ty::get(indexed_ty).sty {
        ty::ty_estr(ty::vstore_slice(r_ptr)) |
        ty::ty_evec(_, ty::vstore_slice(r_ptr)) => {
            rcx.fcx.mk_subr(true, infer::IndexSlice(index_expr.span),
                            r_index_expr, r_ptr);
        }

        _ => {}
    }
}

fn constrain_free_variables(rcx: @mut Rcx,
                            region: ty::Region,
                            expr: @ast::Expr) {
    /*!
     * Make sure that all free variables referenced inside the closure
     * outlive the closure itself.
     */

    let tcx = rcx.fcx.ccx.tcx;
    debug!("constrain_free_variables(%s, %s)",
           region.repr(tcx), expr.repr(tcx));
    for freevar in get_freevars(tcx, expr.id).iter() {
        debug!("freevar def is %?", freevar.def);
        let def = freevar.def;
        let en_region = encl_region_of_def(rcx.fcx, def);
        debug!("en_region = %s", en_region.repr(tcx));
        rcx.fcx.mk_subr(true, infer::FreeVariable(freevar.span),
                        region, en_region);
    }
}

fn constrain_regions_in_type_of_node(
    rcx: @mut Rcx,
    id: ast::NodeId,
    minimum_lifetime: ty::Region,
    origin: infer::SubregionOrigin) -> bool
{
    //! Guarantees that any lifetimes which appear in the type of
    //! the node `id` (after applying adjustments) are valid for at
    //! least `minimum_lifetime`

    let tcx = rcx.fcx.tcx();

    // Try to resolve the type.  If we encounter an error, then typeck
    // is going to fail anyway, so just stop here and let typeck
    // report errors later on in the writeback phase.
    let ty0 = rcx.resolve_node_type(id);
    let adjustment = rcx.fcx.inh.adjustments.find_copy(&id);
    let ty = ty::adjust_ty(tcx, origin.span(), ty0, adjustment);
    debug!("constrain_regions_in_type_of_node(\
            ty=%s, ty0=%s, id=%d, minimum_lifetime=%?, adjustment=%?)",
           ty_to_str(tcx, ty), ty_to_str(tcx, ty0),
           id, minimum_lifetime, adjustment);
    constrain_regions_in_type(rcx, minimum_lifetime, origin, ty)
}

fn constrain_regions_in_type(
    rcx: @mut Rcx,
    minimum_lifetime: ty::Region,
    origin: infer::SubregionOrigin,
    ty: ty::t) -> bool
{
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

    let e = rcx.errors_reported;
    let tcx = rcx.fcx.ccx.tcx;

    debug!("constrain_regions_in_type(minimum_lifetime=%s, ty=%s)",
           region_to_str(tcx, "", false, minimum_lifetime),
           ty_to_str(tcx, ty));

    do relate_nested_regions(tcx, Some(minimum_lifetime), ty) |r_sub, r_sup| {
        debug!("relate(r_sub=%s, r_sup=%s)",
               region_to_str(tcx, "", false, r_sub),
               region_to_str(tcx, "", false, r_sup));

        if r_sup.is_bound() || r_sub.is_bound() {
            // a bound region is one which appears inside an fn type.
            // (e.g., the `&` in `fn(&T)`).  Such regions need not be
            // constrained by `minimum_lifetime` as they are placeholders
            // for regions that are as-yet-unknown.
        } else if r_sub == minimum_lifetime {
            rcx.fcx.mk_subr(
                true, origin,
                r_sub, r_sup);
        } else {
            rcx.fcx.mk_subr(
                true, infer::ReferenceOutlivesReferent(ty, origin.span()),
                r_sub, r_sup);
        }
    }

    return (e == rcx.errors_reported);
}

pub mod guarantor {
    /*!
     * The routines in this module are aiming to deal with the case
     * where a the contents of a borrowed pointer are re-borrowed.
     * Imagine you have a borrowed pointer `b` with lifetime L1 and
     * you have an expression `&*b`.  The result of this borrow will
     * be another borrowed pointer with lifetime L2 (which is an
     * inference variable).  The borrow checker is going to enforce
     * the constraint that L2 < L1, because otherwise you are
     * re-borrowing data for a lifetime larger than the original loan.
     * However, without the routines in this module, the region
     * inferencer would not know of this dependency and thus it might
     * infer the lifetime of L2 to be greater than L1 (issue #3148).
     *
     * There are a number of troublesome scenarios in the tests
     * `region-dependent-*.rs`, but here is one example:
     *
     *     struct Foo { i: int }
     *     struct Bar { foo: Foo  }
     *     fn get_i(x: &'a Bar) -> &'a int {
     *        let foo = &x.foo; // Lifetime L1
     *        &foo.i            // Lifetime L2
     *     }
     *
     * Note that this comes up either with `&` expressions, `ref`
     * bindings, and `autorefs`, which are the three ways to introduce
     * a borrow.
     *
     * The key point here is that when you are borrowing a value that
     * is "guaranteed" by a borrowed pointer, you must link the
     * lifetime of that borrowed pointer (L1, here) to the lifetime of
     * the borrow itself (L2).  What do I mean by "guaranteed" by a
     * borrowed pointer? I mean any data that is reached by first
     * dereferencing a borrowed pointer and then either traversing
     * interior offsets or owned pointers.  We say that the guarantor
     * of such data it the region of the borrowed pointer that was
     * traversed.  This is essentially the same as the ownership
     * relation, except that a borrowed pointer never owns its
     * contents.
     *
     * NB: I really wanted to use the `mem_categorization` code here
     * but I cannot because final type resolution hasn't happened yet,
     * and `mem_categorization` requires that all types be known.
     * So this is very similar logic to what you would find there,
     * but more special purpose.
     */


    use middle::typeck::check::regionck::Rcx;
    use middle::typeck::check::regionck::mk_subregion_due_to_derefence;
    use middle::typeck::infer;
    use middle::ty;
    use syntax::ast;
    use syntax::codemap::Span;
    use util::ppaux::{ty_to_str};

    pub fn for_addr_of(rcx: @mut Rcx, expr: @ast::Expr, base: @ast::Expr) {
        /*!
         * Computes the guarantor for an expression `&base` and then
         * ensures that the lifetime of the resulting pointer is linked
         * to the lifetime of its guarantor (if any).
         */

        debug!("guarantor::for_addr_of(base=?)");

        let guarantor = guarantor(rcx, base);
        link(rcx, expr.span, expr.id, guarantor);
    }

    pub fn for_match(rcx: @mut Rcx, discr: @ast::Expr, arms: &[ast::Arm]) {
        /*!
         * Computes the guarantors for any ref bindings in a match and
         * then ensures that the lifetime of the resulting pointer is
         * linked to the lifetime of its guarantor (if any).
         */

        debug!("regionck::for_match()");
        let discr_guarantor = guarantor(rcx, discr);
        debug!("discr_guarantor=%s", discr_guarantor.repr(rcx.tcx()));
        for arm in arms.iter() {
            for pat in arm.pats.iter() {
                link_ref_bindings_in_pat(rcx, *pat, discr_guarantor);
            }
        }
    }

    pub fn for_autoref(rcx: @mut Rcx,
                       expr: @ast::Expr,
                       autoderefs: uint,
                       autoref: &ty::AutoRef) {
        /*!
         * Computes the guarantor for an expression that has an
         * autoref adjustment and links it to the lifetime of the
         * autoref.  This is only important when auto re-borrowing
         * region pointers.
         */

        debug!("guarantor::for_autoref(autoref=%?)", autoref);

        let mut expr_ct = categorize_unadjusted(rcx, expr);
        debug!("    unadjusted cat=%?", expr_ct.cat);
        expr_ct = apply_autoderefs(
            rcx, expr, autoderefs, expr_ct);

        match *autoref {
            ty::AutoPtr(r, _) => {
                // In this case, we are implicitly adding an `&`.
                maybe_make_subregion(rcx, expr, r, expr_ct.cat.guarantor);
            }

            ty::AutoBorrowVec(r, _) |
            ty::AutoBorrowVecRef(r, _) |
            ty::AutoBorrowFn(r) |
            ty::AutoBorrowObj(r, _) => {
                // In each of these cases, what is being borrowed is
                // not the (autoderef'd) expr itself but rather the
                // contents of the autoderef'd expression (i.e., what
                // the pointer points at).
                maybe_make_subregion(rcx, expr, r,
                                     guarantor_of_deref(&expr_ct.cat));
            }

            ty::AutoUnsafe(_) => {}
        }

        fn maybe_make_subregion(
            rcx: @mut Rcx,
            expr: @ast::Expr,
            sub_region: ty::Region,
            sup_region: Option<ty::Region>)
        {
            for r in sup_region.iter() {
                rcx.fcx.mk_subr(true, infer::Reborrow(expr.span),
                                sub_region, *r);
            }
        }
    }

    pub fn for_by_ref(rcx: @mut Rcx,
                      expr: @ast::Expr,
                      callee_scope: ast::NodeId) {
        /*!
         * Computes the guarantor for cases where the `expr` is
         * being passed by implicit reference and must outlive
         * `callee_scope`.
         */

        let tcx = rcx.tcx();
        debug!("guarantor::for_by_ref(expr=%s, callee_scope=%?)",
               expr.repr(tcx), callee_scope);
        let expr_cat = categorize(rcx, expr);
        debug!("guarantor::for_by_ref(expr=%?, callee_scope=%?) category=%?",
               expr.id, callee_scope, expr_cat);
        let minimum_lifetime = ty::re_scope(callee_scope);
        for guarantor in expr_cat.guarantor.iter() {
            mk_subregion_due_to_derefence(rcx, expr.span,
                                          minimum_lifetime, *guarantor);
        }
    }

    fn link(
        rcx: @mut Rcx,
        span: Span,
        id: ast::NodeId,
        guarantor: Option<ty::Region>) {
        /*!
         *
         * Links the lifetime of the borrowed pointer resulting from a borrow
         * to the lifetime of its guarantor (if any).
         */

        debug!("link(id=%?, guarantor=%?)", id, guarantor);

        let bound = match guarantor {
            None => {
                // If guarantor is None, then the value being borrowed
                // is not guaranteed by a region pointer, so there are
                // no lifetimes to link.
                return;
            }
            Some(r) => { r }
        };

        // this routine is used for the result of ref bindings and &
        // expressions, both of which always yield a region variable, so
        // mk_subr should never fail.
        let rptr_ty = rcx.resolve_node_type(id);
        if !ty::type_is_bot(rptr_ty) {
            let tcx = rcx.fcx.ccx.tcx;
            debug!("rptr_ty=%s", ty_to_str(tcx, rptr_ty));
            let r = ty::ty_region(tcx, span, rptr_ty);
            rcx.fcx.mk_subr(true, infer::Reborrow(span), r, bound);
        }
    }

    /// Categorizes types based on what kind of pointer they are.
    /// Note that we don't bother to distinguish between rptrs (&T)
    /// and slices (&[T], &str)---they are all just `BorrowedPointer`.
    enum PointerCategorization {
        NotPointer,
        OwnedPointer,
        BorrowedPointer(ty::Region),
        OtherPointer
    }

    /// Guarantor of an expression paired with the
    /// PointerCategorization` of its type.
    struct ExprCategorization {
        guarantor: Option<ty::Region>,
        pointer: PointerCategorization
    }

    /// ExprCategorization paired with the full type of the expr
    struct ExprCategorizationType {
        cat: ExprCategorization,
        ty: ty::t
    }

    fn guarantor(rcx: @mut Rcx, expr: @ast::Expr) -> Option<ty::Region> {
        /*!
         *
         * Computes the guarantor of `expr`, or None if `expr` is
         * not guaranteed by any region.  Here `expr` is some expression
         * whose address is being taken (e.g., there is an expression
         * `&expr`).
         */

        debug!("guarantor()");
        match expr.node {
            ast::ExprUnary(_, ast::UnDeref, b) => {
                let cat = categorize(rcx, b);
                guarantor_of_deref(&cat)
            }
            ast::ExprField(b, _, _) => {
                categorize(rcx, b).guarantor
            }
            ast::ExprIndex(_, b, _) => {
                let cat = categorize(rcx, b);
                guarantor_of_deref(&cat)
            }

            ast::ExprParen(e) => {
                guarantor(rcx, e)
            }

            ast::ExprPath(*) | ast::ExprSelf => {
                // Either a variable or constant and hence resides
                // in constant memory or on the stack frame.  Either way,
                // not guaranteed by a region pointer.
                None
            }

            // All of these expressions are rvalues and hence their
            // value is not guaranteed by a region pointer.
            ast::ExprInlineAsm(*) |
            ast::ExprMac(*) |
            ast::ExprLit(_) |
            ast::ExprUnary(*) |
            ast::ExprAddrOf(*) |
            ast::ExprBinary(*) |
            ast::ExprVstore(*) |
            ast::ExprBreak(*) |
            ast::ExprAgain(*) |
            ast::ExprRet(*) |
            ast::ExprLogLevel |
            ast::ExprWhile(*) |
            ast::ExprLoop(*) |
            ast::ExprAssign(*) |
            ast::ExprAssignOp(*) |
            ast::ExprCast(*) |
            ast::ExprCall(*) |
            ast::ExprMethodCall(*) |
            ast::ExprStruct(*) |
            ast::ExprTup(*) |
            ast::ExprIf(*) |
            ast::ExprMatch(*) |
            ast::ExprFnBlock(*) |
            ast::ExprDoBody(*) |
            ast::ExprBlock(*) |
            ast::ExprRepeat(*) |
            ast::ExprVec(*) => {
                assert!(!ty::expr_is_lval(
                    rcx.fcx.tcx(), rcx.fcx.inh.method_map, expr));
                None
            }
            ast::ExprForLoop(*) => fail!("non-desugared expr_for_loop"),
        }
    }

    fn categorize(rcx: @mut Rcx, expr: @ast::Expr) -> ExprCategorization {
        debug!("categorize()");

        let mut expr_ct = categorize_unadjusted(rcx, expr);
        debug!("before adjustments, cat=%?", expr_ct.cat);

        match rcx.fcx.inh.adjustments.find(&expr.id) {
            Some(&@ty::AutoAddEnv(*)) => {
                // This is basically an rvalue, not a pointer, no regions
                // involved.
                expr_ct.cat = ExprCategorization {
                    guarantor: None,
                    pointer: NotPointer
                };
            }

            Some(&@ty::AutoDerefRef(ref adjustment)) => {
                debug!("adjustment=%?", adjustment);

                expr_ct = apply_autoderefs(
                    rcx, expr, adjustment.autoderefs, expr_ct);

                match adjustment.autoref {
                    None => {
                    }
                    Some(ty::AutoUnsafe(_)) => {
                        expr_ct.cat.guarantor = None;
                        expr_ct.cat.pointer = OtherPointer;
                        debug!("autoref, cat=%?", expr_ct.cat);
                    }
                    Some(ty::AutoPtr(r, _)) |
                    Some(ty::AutoBorrowVec(r, _)) |
                    Some(ty::AutoBorrowVecRef(r, _)) |
                    Some(ty::AutoBorrowFn(r)) |
                    Some(ty::AutoBorrowObj(r, _)) => {
                        // If there is an autoref, then the result of this
                        // expression will be some sort of borrowed pointer.
                        expr_ct.cat.guarantor = None;
                        expr_ct.cat.pointer = BorrowedPointer(r);
                        debug!("autoref, cat=%?", expr_ct.cat);
                    }
                }
            }

            None => {}
        }

        debug!("result=%?", expr_ct.cat);
        return expr_ct.cat;
    }

    fn categorize_unadjusted(rcx: @mut Rcx,
                             expr: @ast::Expr)
                          -> ExprCategorizationType {
        debug!("categorize_unadjusted()");

        let guarantor = {
            if rcx.fcx.inh.method_map.contains_key(&expr.id) {
                None
            } else {
                guarantor(rcx, expr)
            }
        };

        let expr_ty = rcx.resolve_node_type(expr.id);
        ExprCategorizationType {
            cat: ExprCategorization {
                guarantor: guarantor,
                pointer: pointer_categorize(expr_ty)
            },
            ty: expr_ty
        }
    }

    fn apply_autoderefs(
        rcx: @mut Rcx,
        expr: @ast::Expr,
        autoderefs: uint,
        ct: ExprCategorizationType)
     -> ExprCategorizationType {
        let mut ct = ct;
        let tcx = rcx.fcx.ccx.tcx;

        if (ty::type_is_error(ct.ty)) {
            ct.cat.pointer = NotPointer;
            return ct;
        }

        for _ in range(0u, autoderefs) {
            ct.cat.guarantor = guarantor_of_deref(&ct.cat);

            match ty::deref(tcx, ct.ty, true) {
                Some(mt) => {
                    ct.ty = mt.ty;
                    ct.cat.pointer = pointer_categorize(ct.ty);
                }
                None => {
                    tcx.sess.span_bug(
                        expr.span,
                        fmt!("Autoderef but type not derefable: %s",
                             ty_to_str(tcx, ct.ty)));
                }
            }

            debug!("autoderef, cat=%?", ct.cat);
        }
        return ct;
    }

    fn pointer_categorize(ty: ty::t) -> PointerCategorization {
        match ty::get(ty).sty {
            ty::ty_rptr(r, _) |
            ty::ty_evec(_, ty::vstore_slice(r)) |
            ty::ty_trait(_, _, ty::RegionTraitStore(r), _, _) |
            ty::ty_estr(ty::vstore_slice(r)) => {
                BorrowedPointer(r)
            }
            ty::ty_uniq(*) |
            ty::ty_estr(ty::vstore_uniq) |
            ty::ty_trait(_, _, ty::UniqTraitStore, _, _) |
            ty::ty_evec(_, ty::vstore_uniq) => {
                OwnedPointer
            }
            ty::ty_box(*) |
            ty::ty_ptr(*) |
            ty::ty_evec(_, ty::vstore_box) |
            ty::ty_trait(_, _, ty::BoxTraitStore, _, _) |
            ty::ty_estr(ty::vstore_box) => {
                OtherPointer
            }
            ty::ty_closure(ref closure_ty) => {
                match closure_ty.sigil {
                    ast::BorrowedSigil => BorrowedPointer(closure_ty.region),
                    ast::OwnedSigil => OwnedPointer,
                    ast::ManagedSigil => OtherPointer,
                }
            }
            _ => {
                NotPointer
            }
        }
    }

    fn guarantor_of_deref(cat: &ExprCategorization) -> Option<ty::Region> {
        match cat.pointer {
            NotPointer => cat.guarantor,
            BorrowedPointer(r) => Some(r),
            OwnedPointer => cat.guarantor,
            OtherPointer => None
        }
    }

    fn link_ref_bindings_in_pat(
        rcx: @mut Rcx,
        pat: @ast::Pat,
        guarantor: Option<ty::Region>) {
        /*!
         *
         * Descends through the pattern, tracking the guarantor
         * of the value being matched.  When a ref binding is encountered,
         * links the lifetime of that ref binding to the lifetime of
         * the guarantor.  We begin with the guarantor of the
         * discriminant but of course as we go we may pass through
         * other pointers.
         */

        debug!("link_ref_bindings_in_pat(pat=%s, guarantor=%?)",
               rcx.fcx.pat_to_str(pat), guarantor);

        match pat.node {
            ast::PatWild => {}
            ast::PatIdent(ast::BindByRef(_), _, opt_p) => {
                link(rcx, pat.span, pat.id, guarantor);

                for p in opt_p.iter() {
                    link_ref_bindings_in_pat(rcx, *p, guarantor);
                }
            }
            ast::PatIdent(_, _, opt_p) => {
                for p in opt_p.iter() {
                    link_ref_bindings_in_pat(rcx, *p, guarantor);
                }
            }
            ast::PatEnum(_, None) => {}
            ast::PatEnum(_, Some(ref pats)) => {
                link_ref_bindings_in_pats(rcx, pats, guarantor);
            }
            ast::PatStruct(_, ref fpats, _) => {
                for fpat in fpats.iter() {
                    link_ref_bindings_in_pat(rcx, fpat.pat, guarantor);
                }
            }
            ast::PatTup(ref ps) => {
                link_ref_bindings_in_pats(rcx, ps, guarantor)
            }
            ast::PatBox(p) => {
                link_ref_bindings_in_pat(rcx, p, None)
            }
            ast::PatUniq(p) => {
                link_ref_bindings_in_pat(rcx, p, guarantor)
            }
            ast::PatRegion(p) => {
                let rptr_ty = rcx.resolve_node_type(pat.id);
                let r = ty::ty_region(rcx.fcx.tcx(), pat.span, rptr_ty);
                link_ref_bindings_in_pat(rcx, p, Some(r));
            }
            ast::PatLit(*) => {}
            ast::PatRange(*) => {}
            ast::PatVec(ref before, ref slice, ref after) => {
                let vec_ty = rcx.resolve_node_type(pat.id);
                let vstore = ty::ty_vstore(vec_ty);
                let guarantor1 = match vstore {
                    ty::vstore_fixed(_) | ty::vstore_uniq => guarantor,
                    ty::vstore_slice(r) => Some(r),
                    ty::vstore_box => None
                };

                link_ref_bindings_in_pats(rcx, before, guarantor1);
                for &p in slice.iter() {
                    link_ref_bindings_in_pat(rcx, p, guarantor);
                }
                link_ref_bindings_in_pats(rcx, after, guarantor1);
            }
        }
    }

    fn link_ref_bindings_in_pats(rcx: @mut Rcx,
                                 pats: &~[@ast::Pat],
                                 guarantor: Option<ty::Region>) {
        for pat in pats.iter() {
            link_ref_bindings_in_pat(rcx, *pat, guarantor);
        }
    }

}
