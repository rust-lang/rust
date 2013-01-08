// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*

The region check is a final pass that runs over the AST after we have
inferred the type constraints but before we have actually finalized
the types.  Its purpose is to embed some final region constraints.
The reason that this is not done earlier is that sometimes we don't
know whether a given type will be a region pointer or not until this
phase.

In particular, we ensure that, if the type of an expression or
variable is `&r/T`, then the expression or variable must occur within
the region scope `r`.  Note that in some cases `r` may still be a
region variable, so this gives us a chance to influence the value for
`r` that we infer to ensure we choose a value large enough to enclose
all uses.  There is a lengthy comment in visit_node() that explains
this point a bit better.

*/

use core::prelude::*;

use middle::freevars::get_freevars;
use middle::pat_util::pat_bindings;
use middle::ty::{encl_region, re_scope};
use middle::ty::{ty_fn_proto, vstore_box, vstore_fixed, vstore_slice};
use middle::ty::{vstore_uniq};
use middle::ty;
use middle::typeck::check::fn_ctxt;
use middle::typeck::check::lookup_def;
use middle::typeck::infer::{fres, resolve_and_force_all_but_regions};
use middle::typeck::infer::{resolve_type};
use util::ppaux::{note_and_explain_region, ty_to_str};

use core::result;
use syntax::ast::{ProtoBare, ProtoBox, ProtoUniq, ProtoBorrowed};
use syntax::ast::{def_arg, def_binding, def_local, def_self, def_upvar};
use syntax::ast;
use syntax::codemap::span;
use syntax::print::pprust;
use syntax::visit;

enum rcx { rcx_({fcx: @fn_ctxt, mut errors_reported: uint}) }
type rvt = visit::vt<@rcx>;

fn encl_region_of_def(fcx: @fn_ctxt, def: ast::def) -> ty::Region {
    let tcx = fcx.tcx();
    match def {
        def_local(node_id, _) | def_arg(node_id, _) | def_self(node_id, _) |
        def_binding(node_id, _) =>
            return encl_region(tcx, node_id),
        def_upvar(_, subdef, closure_id, body_id) => {
            match ty_fn_proto(fcx.node_ty(closure_id)) {
                ProtoBare => tcx.sess.bug(~"ProtoBare with upvars?!"),
                ProtoBorrowed => encl_region_of_def(fcx, *subdef),
                ProtoBox | ProtoUniq => re_scope(body_id)
            }
        }
        _ => {
            tcx.sess.bug(fmt!("unexpected def in encl_region_of_def: %?",
                              def))
        }
    }
}

impl @rcx {
    /// Try to resolve the type for the given node.
    ///
    /// Note one important point: we do not attempt to resolve *region
    /// variables* here.  This is because regionck is essentially adding
    /// constraints to those region variables and so may yet influence
    /// how they are resolved.
    ///
    /// Consider this silly example:
    ///
    ///     fn borrow(x: &int) -> &int {x}
    ///     fn foo(x: @int) -> int {  /* block: B */
    ///         let b = borrow(x);    /* region: <R0> */
    ///         *b
    ///     }
    ///
    /// Here, the region of `b` will be `<R0>`.  `<R0>` is constrainted
    /// to be some subregion of the block B and some superregion of
    /// the call.  If we forced it now, we'd choose the smaller region
    /// (the call).  But that would make the *b illegal.  Since we don't
    /// resolve, the type of b will be `&<R0>.int` and then `*b` will require
    /// that `<R0>` be bigger than the let and the `*b` expression, so we
    /// will effectively resolve `<R0>` to be the block B.
    fn resolve_type(unresolved_ty: ty::t) -> fres<ty::t> {
        resolve_type(self.fcx.infcx(), unresolved_ty,
                     resolve_and_force_all_but_regions)
    }

    /// Try to resolve the type for the given node.
    fn resolve_node_type(id: ast::node_id) -> fres<ty::t> {
        self.resolve_type(self.fcx.node_ty(id))
    }
}

fn regionck_expr(fcx: @fn_ctxt, e: @ast::expr) {
    let rcx = rcx_({fcx:fcx, mut errors_reported: 0});
    let v = regionck_visitor();
    (v.visit_expr)(e, @(move rcx), v);
    fcx.infcx().resolve_regions();
}

fn regionck_fn(fcx: @fn_ctxt,
               _decl: ast::fn_decl,
               blk: ast::blk) {
    let rcx = rcx_({fcx:fcx, mut errors_reported: 0});
    let v = regionck_visitor();
    (v.visit_block)(blk, @(move rcx), v);
    fcx.infcx().resolve_regions();
}

fn regionck_visitor() -> rvt {
    visit::mk_vt(@{visit_item: visit_item,
                   visit_stmt: visit_stmt,
                   visit_expr: visit_expr,
                   visit_block: visit_block,
                   visit_local: visit_local,
                   .. *visit::default_visitor()})
}

fn visit_item(_item: @ast::item, &&_rcx: @rcx, _v: rvt) {
    // Ignore items
}

fn visit_local(l: @ast::local, &&rcx: @rcx, v: rvt) {
    // Check to make sure that the regions in all local variables are
    // within scope.
    //
    // Note: we do this here rather than in visit_pat because we do
    // not wish to constrain the regions in *patterns* in quite the
    // same way.  `visit_node()` guarantees that the region encloses
    // the node in question, which ultimately constraints the regions
    // in patterns to enclose the match expression as a whole.  But we
    // want them to enclose the *arm*.  However, regions in patterns
    // must either derive from the discriminant or a ref pattern: in
    // the case of the discriminant, the regions will be constrained
    // when the type of the discriminant is checked.  In the case of a
    // ref pattern, the variable is created with a suitable lower
    // bound.
    let e = rcx.errors_reported;
    (v.visit_pat)(l.node.pat, rcx, v);
    let def_map = rcx.fcx.ccx.tcx.def_map;
    do pat_bindings(def_map, l.node.pat) |_bm, id, sp, _path| {
        visit_node(id, sp, rcx);
    }
    if e != rcx.errors_reported {
        return; // if decl has errors, skip initializer expr
    }

    (v.visit_ty)(l.node.ty, rcx, v);
    for l.node.init.each |i| {
        (v.visit_expr)(*i, rcx, v);
    }
}

fn visit_block(b: ast::blk, &&rcx: @rcx, v: rvt) {
    visit::visit_block(b, rcx, v);
}

fn visit_expr(expr: @ast::expr, &&rcx: @rcx, v: rvt) {
    debug!("visit_expr(e=%s)",
           pprust::expr_to_str(expr, rcx.fcx.tcx().sess.intr()));

    match /*bad*/copy expr.node {
        ast::expr_path(*) => {
            // Avoid checking the use of local variables, as we
            // already check their definitions.  The def'n always
            // encloses the use.  So if the def'n is enclosed by the
            // region, then the uses will also be enclosed (and
            // otherwise, an error will have been reported at the
            // def'n site).
            match lookup_def(rcx.fcx, expr.span, expr.id) {
                ast::def_local(*) | ast::def_arg(*) |
                ast::def_upvar(*) => return,
                _ => ()
            }
        }

        ast::expr_call(callee, args, _) => {
            // Check for a.b() where b is a method.  Ensure that
            // any types in the callee are valid for the entire
            // method call.

            // FIXME(#3387)--we should really invoke
            // `constrain_auto_ref()` on all exprs.  But that causes a
            // lot of spurious errors because of how the region
            // hierarchy is setup.
            if rcx.fcx.ccx.method_map.contains_key(callee.id) {
                match callee.node {
                    ast::expr_field(base, _, _) => {
                        constrain_auto_ref(rcx, base);
                    }
                    _ => {
                        // This can happen if you have code like
                        // (x[0])() where `x[0]` is overloaded.  Just
                        // ignore it.
                    }
                }
            } else {
                constrain_auto_ref(rcx, callee);
            }

            for args.each |arg| {
                constrain_auto_ref(rcx, *arg);
            }
        }

        ast::expr_method_call(rcvr, _, _, args, _) => {
            // Check for a.b() where b is a method.  Ensure that
            // any types in the callee are valid for the entire
            // method call.

            constrain_auto_ref(rcx, rcvr);
            for args.each |arg| {
                constrain_auto_ref(rcx, *arg);
            }
        }

        ast::expr_cast(source, _) => {
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
            match rcx.resolve_node_type(expr.id) {
                result::Err(_) => { return; /*typeck will fail anyhow*/ }
                result::Ok(target_ty) => {
                    match ty::get(target_ty).sty {
                        ty::ty_trait(_, _, vstore_slice(trait_region)) => {
                            let source_ty = rcx.fcx.expr_ty(source);
                            constrain_regions_in_type(rcx, trait_region,
                                                      expr.span, source_ty);
                        }
                        _ => ()
                    }
                }
            };
        }

        ast::expr_addr_of(*) => {
            // FIXME(#3148) -- in some cases, we need to capture a
            // dependency between the regions found in operand the
            // resulting region type.  See #3148 for more details.
        }

        ast::expr_fn(*) | ast::expr_fn_block(*) => {
            match rcx.resolve_node_type(expr.id) {
                result::Err(_) => return, // Typechecking will fail anyhow.
                result::Ok(function_type) => {
                    match ty::get(function_type).sty {
                        ty::ty_fn(ref fn_ty) => {
                            if fn_ty.meta.proto == ast::ProtoBorrowed {
                                constrain_free_variables(
                                    rcx, fn_ty.meta.region, expr);
                            }
                        }
                        _ => ()
                    }
                }
            }
        }

        _ => ()
    }

    if !visit_node(expr.id, expr.span, rcx) { return; }
    visit::visit_expr(expr, rcx, v);
}

fn visit_stmt(s: @ast::stmt, &&rcx: @rcx, v: rvt) {
    visit::visit_stmt(s, rcx, v);
}

fn visit_node(id: ast::node_id, span: span, rcx: @rcx) -> bool {
    /*!
     *
     * checks the type of the node `id` and reports an error if it
     * references a region that is not in scope for that node.
     * Returns false if an error is reported; this is used to cause us
     * to cut off region checking for that subtree to avoid reporting
     * tons of errors. */

    let fcx = rcx.fcx;

    // find the region where this expr evaluation is taking place
    let tcx = fcx.ccx.tcx;
    let encl_region = ty::encl_region(tcx, id);

    // Otherwise, look at the type and see if it is a region pointer.
    constrain_regions_in_type_of_node(rcx, id, encl_region, span)
}

fn constrain_auto_ref(
    rcx: @rcx,
    expr: @ast::expr)
{
    /*!
     *
     * If `expr` is auto-ref'd (e.g., as part of a borrow), then this
     * function ensures that the lifetime of the resulting borrowed
     * ptr includes at least the expression `expr`. */

    debug!("constrain_auto_ref(expr=%s)", rcx.fcx.expr_to_str(expr));

    let adjustment = rcx.fcx.inh.adjustments.find(expr.id);
    let region = match adjustment {
        Some(@{autoref: Some(ref auto_ref), _}) => auto_ref.region,
        _ => { return; }
    };

    let tcx = rcx.fcx.tcx();
    let encl_region = ty::encl_region(tcx, expr.id);
    match rcx.fcx.mk_subr(true, expr.span, encl_region, region) {
        result::Ok(()) => {}
        result::Err(_) => {
            // In practice, this cannot happen: `region` is always a
            // region variable, and constraints on region variables
            // are collected and then resolved later.  However, I
            // included the span_err() here (rather than, say,
            // span_bug()) because it seemed more future-proof: if,
            // for some reason, the code were to change so that in
            // some cases `region` is not a region variable, then
            // reporting an error would be the correct path.
            tcx.sess.span_err(
                expr.span,
                ~"lifetime of borrowed pointer does not include \
                  the expression being borrowed");
            note_and_explain_region(
                tcx,
                ~"lifetime of the borrowed pointer is",
                region,
                ~"");
            rcx.errors_reported += 1;
        }
    }
}

fn constrain_free_variables(
    rcx: @rcx,
    region: ty::Region,
    expr: @ast::expr)
{
    /*!
     *
     * Make sure that all free variables referenced inside the closure
     * outlive the closure itself. */

    let tcx = rcx.fcx.ccx.tcx;
    for get_freevars(tcx, expr.id).each |freevar| {
        debug!("freevar def is %?", freevar.def);
        let def = freevar.def;
        let en_region = encl_region_of_def(rcx.fcx, def);
        match rcx.fcx.mk_subr(true, freevar.span,
                              region, en_region) {
          result::Ok(()) => {}
          result::Err(_) => {
            tcx.sess.span_err(
                freevar.span,
                ~"captured variable does not outlive the enclosing closure");
            note_and_explain_region(
                tcx,
                ~"captured variable is valid for ",
                en_region,
                ~"");
            note_and_explain_region(
                tcx,
                ~"closure is valid for ",
                region,
                ~"");
          }
        }
    }
}

fn constrain_regions_in_type_of_node(
    rcx: @rcx,
    id: ast::node_id,
    encl_region: ty::Region,
    span: span) -> bool
{
    let tcx = rcx.fcx.tcx();

    // Try to resolve the type.  If we encounter an error, then typeck
    // is going to fail anyway, so just stop here and let typeck
    // report errors later on in the writeback phase.
    let ty = match rcx.resolve_node_type(id) {
      result::Err(_) => return true,
      result::Ok(ty) => ty
    };

    debug!("constrain_regions_in_type_of_node(\
            ty=%s, id=%d, encl_region=%?)",
           ty_to_str(tcx, ty), id, encl_region);

    constrain_regions_in_type(rcx, encl_region, span, ty)
}

fn constrain_regions_in_type(
    rcx: @rcx,
    encl_region: ty::Region,
    span: span,
    ty: ty::t) -> bool
{
    /*!
     *
     * Requires that any regions which appear in `ty` must be
     * superregions of `encl_region`.  This prevents regions from
     * being used outside of the block in which they are valid.
     * Recall that regions represent blocks of code or expressions:
     * this requirement basically says "any place that uses or may use
     * a region R must be within the block of code that R corresponds
     * to." */

    let e = rcx.errors_reported;
    ty::walk_regions_and_ty(
        rcx.fcx.ccx.tcx, ty,
        |r| constrain_region(rcx, encl_region, span, r),
        |t| ty::type_has_regions(t));
    return (e == rcx.errors_reported);

    fn constrain_region(rcx: @rcx,
                        encl_region: ty::Region,
                        span: span,
                        region: ty::Region) {
        let tcx = rcx.fcx.ccx.tcx;

        debug!("constrain_region(encl_region=%?, region=%?)",
               encl_region, region);

        match region {
          ty::re_bound(_) => {
            // a bound region is one which appears inside an fn type.
            // (e.g., the `&` in `fn(&T)`).  Such regions need not be
            // constrained by `encl_region` as they are placeholders
            // for regions that are as-yet-unknown.
            return;
          }
          _ => ()
        }

        match rcx.fcx.mk_subr(true, span, encl_region, region) {
          result::Err(_) => {
            tcx.sess.span_err(
                span,
                fmt!("reference is not valid outside of its lifetime"));
            note_and_explain_region(
                tcx,
                ~"the reference is only valid for ",
                region,
                ~"");
            rcx.errors_reported += 1u;
          }
          result::Ok(()) => {
          }
        }
    }
}
