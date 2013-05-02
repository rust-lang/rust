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
variable is `&'r T`, then the expression or variable must occur within
the region scope `r`.  Note that in some cases `r` may still be a
region variable, so this gives us a chance to influence the value for
`r` that we infer to ensure we choose a value large enough to enclose
all uses.  There is a lengthy comment in visit_node() that explains
this point a bit better.

*/

use middle::freevars::get_freevars;
use middle::pat_util::pat_bindings;
use middle::ty::{re_scope};
use middle::ty;
use middle::typeck::check::FnCtxt;
use middle::typeck::check::lookup_def;
use middle::typeck::check::regionmanip::relate_nested_regions;
use middle::typeck::infer::resolve_and_force_all_but_regions;
use middle::typeck::infer::resolve_type;
use util::ppaux::{note_and_explain_region, ty_to_str,
                  region_to_str};

use syntax::ast::{ManagedSigil, OwnedSigil, BorrowedSigil};
use syntax::ast::{def_arg, def_binding, def_local, def_self, def_upvar};
use syntax::ast;
use syntax::codemap::span;
use syntax::visit;

pub struct Rcx {
    fcx: @mut FnCtxt,
    errors_reported: uint
}

pub type rvt = visit::vt<@mut Rcx>;

fn encl_region_of_def(fcx: @mut FnCtxt, def: ast::def) -> ty::Region {
    let tcx = fcx.tcx();
    match def {
        def_local(node_id, _) | def_arg(node_id, _) |
        def_self(node_id, _) | def_binding(node_id, _) => {
            tcx.region_maps.encl_region(node_id)
        }
        def_upvar(_, subdef, closure_id, body_id) => {
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

pub impl Rcx {
    fn resolve_type(@mut self, unresolved_ty: ty::t) -> ty::t {
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
    fn resolve_node_type(@mut self, id: ast::node_id) -> ty::t {
        self.resolve_type(self.fcx.node_ty(id))
    }

    /// Try to resolve the type for the given node.
    fn resolve_expr_type_adjusted(@mut self, expr: @ast::expr) -> ty::t {
        let ty_unadjusted = self.resolve_node_type(expr.id);
        if ty::type_is_error(ty_unadjusted) || ty::type_is_bot(ty_unadjusted) {
            ty_unadjusted
        } else {
            let tcx = self.fcx.tcx();
            let adjustments = self.fcx.inh.adjustments;
            match adjustments.find(&expr.id) {
                None => ty_unadjusted,
                Some(&adjustment) => {
                    // FIXME(#3850) --- avoid region scoping errors
                    ty::adjust_ty(tcx, expr.span, ty_unadjusted, Some(&adjustment))
                }
            }
        }
    }
}

pub fn regionck_expr(fcx: @mut FnCtxt, e: @ast::expr) {
    let rcx = @mut Rcx { fcx: fcx, errors_reported: 0 };
    let v = regionck_visitor();
    (v.visit_expr)(e, rcx, v);
    fcx.infcx().resolve_regions();
}

pub fn regionck_fn(fcx: @mut FnCtxt, blk: &ast::blk) {
    let rcx = @mut Rcx { fcx: fcx, errors_reported: 0 };
    let v = regionck_visitor();
    (v.visit_block)(blk, rcx, v);
    fcx.infcx().resolve_regions();
}

fn regionck_visitor() -> rvt {
    visit::mk_vt(@visit::Visitor {visit_item: visit_item,
                                  visit_stmt: visit_stmt,
                                  visit_expr: visit_expr,
                                  visit_block: visit_block,
                                  visit_local: visit_local,
                                  .. *visit::default_visitor()})
}

fn visit_item(_item: @ast::item, _rcx: @mut Rcx, _v: rvt) {
    // Ignore items
}

fn visit_local(l: @ast::local, rcx: @mut Rcx, v: rvt) {
    // Check to make sure that the regions in all local variables are
    // within scope.
    //
    // Note: we do this here rather than in visit_pat because we do
    // not wish to constrain the regions in *patterns* in quite the
    // same way.  `visit_node()` guarantees that the region encloses
    // the node in question, which ultimately constrains the regions
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

fn visit_block(b: &ast::blk, rcx: @mut Rcx, v: rvt) {
    visit::visit_block(b, rcx, v);
}

fn visit_expr(expr: @ast::expr, rcx: @mut Rcx, v: rvt) {
    debug!("regionck::visit_expr(e=%s)", rcx.fcx.expr_to_str(expr));

    for rcx.fcx.inh.adjustments.find(&expr.id).each |&adjustment| {
        debug!("adjustment=%?", adjustment);
        match *adjustment {
            @ty::AutoDerefRef(
                ty::AutoDerefRef {autoderefs: autoderefs, autoref: opt_autoref}) =>
            {
                let expr_ty = rcx.resolve_node_type(expr.id);
                constrain_derefs(rcx, expr, autoderefs, expr_ty);
                for opt_autoref.each |autoref| {
                    guarantor::for_autoref(rcx, expr, autoderefs, autoref);
                }
            }
            _ => {}
        }
    }

    match expr.node {
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

        ast::expr_call(callee, ref args, _) => {
            // Check for a.b() where b is a method.  Ensure that
            // any types in the callee are valid for the entire
            // method call.

            // FIXME(#3387)--we should really invoke
            // `constrain_auto_ref()` on all exprs.  But that causes a
            // lot of spurious errors because of how the region
            // hierarchy is setup.
            if rcx.fcx.inh.method_map.contains_key(&callee.id) {
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

        ast::expr_method_call(rcvr, _, _, ref args, _) => {
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
            let target_ty = rcx.resolve_node_type(expr.id);
            match ty::get(target_ty).sty {
                ty::ty_trait(_, _, ty::RegionTraitStore(trait_region), _) => {
                    let source_ty = rcx.fcx.expr_ty(source);
                    constrain_regions_in_type(rcx, trait_region,
                                              expr.span, source_ty);
                }
                _ => ()
            }
        }

        ast::expr_index(vec_expr, _) => {
            let vec_type = rcx.resolve_expr_type_adjusted(vec_expr);
            constrain_index(rcx, expr, vec_type);
        }

        ast::expr_unary(ast::deref, base) => {
            let base_ty = rcx.resolve_node_type(base.id);
            constrain_derefs(rcx, expr, 1, base_ty);
        }

        ast::expr_addr_of(_, base) => {
            guarantor::for_addr_of(rcx, expr, base);
        }

        ast::expr_match(discr, ref arms) => {
            guarantor::for_match(rcx, discr, *arms);
        }

        ast::expr_fn_block(*) => {
            let function_type = rcx.resolve_node_type(expr.id);
            match ty::get(function_type).sty {
                ty::ty_closure(ty::ClosureTy {sigil: ast::BorrowedSigil,
                                              region: region, _}) => {
                    constrain_free_variables(rcx, region, expr);
                }
                _ => ()
            }
        }

        _ => ()
    }

    if !visit_node(expr.id, expr.span, rcx) { return; }
    visit::visit_expr(expr, rcx, v);
}

fn visit_stmt(s: @ast::stmt, rcx: @mut Rcx, v: rvt) {
    visit::visit_stmt(s, rcx, v);
}

fn visit_node(id: ast::node_id, span: span, rcx: @mut Rcx) -> bool {
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
    let encl_region = match tcx.region_maps.opt_encl_scope(id) {
        None => ty::re_static,
        Some(r) => ty::re_scope(r)
    };

    // Otherwise, look at the type and see if it is a region pointer.
    constrain_regions_in_type_of_node(rcx, id, encl_region, span)
}

fn encl_region_or_static(rcx: @mut Rcx, expr: @ast::expr) -> ty::Region {
    // FIXME(#3850) --- interactions with modes compel overly large granularity
    // that is, we would probably prefer to just return re_scope(expr.id)
    // here but we cannot just yet.

    let tcx = rcx.fcx.tcx();
    match tcx.region_maps.opt_encl_scope(expr.id) {
        Some(s) => ty::re_scope(s),
        None => ty::re_static // occurs in constants
    }
}

fn constrain_derefs(rcx: @mut Rcx,
                    deref_expr: @ast::expr,
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
    let r_deref_expr = encl_region_or_static(rcx, deref_expr);
    for uint::range(0, derefs) |i| {
        debug!("constrain_derefs(deref_expr=%s, derefd_ty=%s, derefs=%?/%?",
               rcx.fcx.expr_to_str(deref_expr),
               rcx.fcx.infcx().ty_to_str(derefd_ty),
               i, derefs);

        match ty::get(derefd_ty).sty {
            ty::ty_rptr(r_ptr, _) => {
                match rcx.fcx.mk_subr(true, deref_expr.span, r_deref_expr, r_ptr) {
                    result::Ok(*) => {}
                    result::Err(*) => {
                        tcx.sess.span_err(
                            deref_expr.span,
                            fmt!("dereference of reference outside its lifetime"));
                        note_and_explain_region(
                            tcx,
                            "the reference is only valid for ",
                            r_ptr,
                            "");
                    }
                }
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

fn constrain_index(rcx: @mut Rcx,
                   index_expr: @ast::expr,
                   indexed_ty: ty::t)
{
    /*!
     * Invoked on any index expression that occurs.  Checks that if
     * this is a slice being indexed, the lifetime of the pointer
     * includes the deref expr.
     */

    let tcx = rcx.fcx.tcx();

    debug!("constrain_index(index_expr=%s, indexed_ty=%s",
           rcx.fcx.expr_to_str(index_expr),
           rcx.fcx.infcx().ty_to_str(indexed_ty));

    let r_index_expr = encl_region_or_static(rcx, index_expr);
    match ty::get(indexed_ty).sty {
        ty::ty_estr(ty::vstore_slice(r_ptr)) |
        ty::ty_evec(_, ty::vstore_slice(r_ptr)) => {
            match rcx.fcx.mk_subr(true, index_expr.span, r_index_expr, r_ptr) {
                result::Ok(*) => {}
                result::Err(*) => {
                    tcx.sess.span_err(
                        index_expr.span,
                        fmt!("index of slice outside its lifetime"));
                    note_and_explain_region(
                        tcx,
                        "the slice is only valid for ",
                        r_ptr,
                        "");
                }
            }
        }

        _ => {}
    }
}

fn constrain_auto_ref(rcx: @mut Rcx, expr: @ast::expr) {
    /*!
     *
     * If `expr` is auto-ref'd (e.g., as part of a borrow), then this
     * function ensures that the lifetime of the resulting borrowed
     * ptr includes at least the expression `expr`. */

    debug!("constrain_auto_ref(expr=%s)", rcx.fcx.expr_to_str(expr));

    let adjustment = rcx.fcx.inh.adjustments.find(&expr.id);
    let region = match adjustment {
        Some(&@ty::AutoDerefRef(
            ty::AutoDerefRef {
                autoref: Some(ref auto_ref), _})) => {
            auto_ref.region
        }
        _ => { return; }
    };

    let tcx = rcx.fcx.tcx();
    let encl_region = tcx.region_maps.encl_region(expr.id);
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
                "lifetime of borrowed pointer does not include \
                 the expression being borrowed");
            note_and_explain_region(
                tcx,
                "lifetime of the borrowed pointer is",
                region,
                "");
            rcx.errors_reported += 1;
        }
    }
}

fn constrain_free_variables(
    rcx: @mut Rcx,
    region: ty::Region,
    expr: @ast::expr) {
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
                "captured variable does not outlive the enclosing closure");
            note_and_explain_region(
                tcx,
                "captured variable is valid for ",
                en_region,
                "");
            note_and_explain_region(
                tcx,
                "closure is valid for ",
                region,
                "");
          }
        }
    }
}

fn constrain_regions_in_type_of_node(
    rcx: @mut Rcx,
    id: ast::node_id,
    encl_region: ty::Region,
    span: span) -> bool
{
    let tcx = rcx.fcx.tcx();

    // Try to resolve the type.  If we encounter an error, then typeck
    // is going to fail anyway, so just stop here and let typeck
    // report errors later on in the writeback phase.
    let ty0 = rcx.resolve_node_type(id);
    let adjustment = rcx.fcx.inh.adjustments.find(&id);
    let ty = ty::adjust_ty(tcx, span, ty0, adjustment);
    debug!("constrain_regions_in_type_of_node(\
            ty=%s, ty0=%s, id=%d, encl_region=%?, adjustment=%?)",
           ty_to_str(tcx, ty), ty_to_str(tcx, ty0),
           id, encl_region, adjustment);
    constrain_regions_in_type(rcx, encl_region, span, ty)
}

fn constrain_regions_in_type(
    rcx: @mut Rcx,
    encl_region: ty::Region,
    span: span,
    ty: ty::t) -> bool
{
    /*!
     *
     * Requires that any regions which appear in `ty` must be
     * superregions of `encl_region`.  Also enforces the constraint
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

    debug!("constrain_regions_in_type(encl_region=%s, ty=%s)",
           region_to_str(tcx, encl_region),
           ty_to_str(tcx, ty));

    do relate_nested_regions(tcx, Some(encl_region), ty) |r_sub, r_sup| {
        debug!("relate(r_sub=%s, r_sup=%s)",
               region_to_str(tcx, r_sub),
               region_to_str(tcx, r_sup));

        if r_sup.is_bound() || r_sub.is_bound() {
            // a bound region is one which appears inside an fn type.
            // (e.g., the `&` in `fn(&T)`).  Such regions need not be
            // constrained by `encl_region` as they are placeholders
            // for regions that are as-yet-unknown.
        } else {
            match rcx.fcx.mk_subr(true, span, r_sub, r_sup) {
                result::Err(_) => {
                    if r_sub == encl_region {
                        tcx.sess.span_err(
                            span,
                            fmt!("reference is not valid outside of its lifetime"));
                        note_and_explain_region(
                            tcx,
                            "the reference is only valid for ",
                            r_sup,
                            "");
                    } else {
                        tcx.sess.span_err(
                            span,
                            fmt!("in type `%s`, pointer has a longer lifetime than \
                                  the data it references",
                                 rcx.fcx.infcx().ty_to_str(ty)));
                        note_and_explain_region(
                            tcx,
                            "the pointer is valid for ",
                            r_sub,
                            "");
                        note_and_explain_region(
                            tcx,
                            "but the referenced data is only valid for ",
                            r_sup,
                            "");
                    }
                    rcx.errors_reported += 1u;
                }
                result::Ok(()) => {
                }
            }
        }
    }

    return (e == rcx.errors_reported);
}

pub mod guarantor {
    /*!
     *
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

    use middle::typeck::check::regionck::{Rcx, infallibly_mk_subr};
    use middle::ty;
    use syntax::ast;
    use syntax::codemap::span;
    use util::ppaux::{ty_to_str};

    pub fn for_addr_of(rcx: @mut Rcx, expr: @ast::expr, base: @ast::expr) {
        /*!
         *
         * Computes the guarantor for an expression `&base` and then
         * ensures that the lifetime of the resulting pointer is linked
         * to the lifetime of its guarantor (if any).
         */

        debug!("guarantor::for_addr_of(base=%s)", rcx.fcx.expr_to_str(base));
        let _i = ::util::common::indenter();

        let guarantor = guarantor(rcx, base);
        link(rcx, expr.span, expr.id, guarantor);
    }

    pub fn for_match(rcx: @mut Rcx, discr: @ast::expr, arms: &[ast::arm]) {
        /*!
         *
         * Computes the guarantors for any ref bindings in a match and
         * then ensures that the lifetime of the resulting pointer is
         * linked to the lifetime of its guarantor (if any).
         */

        let discr_guarantor = guarantor(rcx, discr);
        for arms.each |arm| {
            for arm.pats.each |pat| {
                link_ref_bindings_in_pat(rcx, *pat, discr_guarantor);
            }
        }
    }

    pub fn for_autoref(rcx: @mut Rcx,
                       expr: @ast::expr,
                       autoderefs: uint,
                       autoref: &ty::AutoRef) {
        /*!
         *
         * Computes the guarantor for an expression that has an
         * autoref adjustment and links it to the lifetime of the
         * autoref.  This is only important when auto re-borrowing
         * region pointers.
         */

        debug!("guarantor::for_autoref(expr=%s, autoref=%?)",
               rcx.fcx.expr_to_str(expr), autoref);
        let _i = ::util::common::indenter();

        let mut expr_ct = categorize_unadjusted(rcx, expr);
        debug!("    unadjusted cat=%?", expr_ct.cat);
        expr_ct = apply_autoderefs(
            rcx, expr, autoderefs, expr_ct);

        match autoref.kind {
            ty::AutoPtr => {
                // In this case, we are implicitly adding an `&`.
                maybe_make_subregion(rcx, expr, autoref.region,
                                     expr_ct.cat.guarantor);
            }

            ty::AutoBorrowVec |
            ty::AutoBorrowVecRef |
            ty::AutoBorrowFn => {
                // In each of these cases, what is being borrowed is
                // not the (autoderef'd) expr itself but rather the
                // contents of the autoderef'd expression (i.e., what
                // the pointer points at).
                maybe_make_subregion(rcx, expr, autoref.region,
                                     guarantor_of_deref(&expr_ct.cat));
            }
        }

        fn maybe_make_subregion(
            rcx: @mut Rcx,
            expr: @ast::expr,
            sub_region: ty::Region,
            sup_region: Option<ty::Region>)
        {
            for sup_region.each |r| {
                infallibly_mk_subr(rcx, true, expr.span, sub_region, *r);
            }
        }
    }

    fn link(
        rcx: @mut Rcx,
        span: span,
        id: ast::node_id,
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
        if !ty::type_is_error(rptr_ty) && !ty::type_is_bot(rptr_ty) {
            let tcx = rcx.fcx.ccx.tcx;
            debug!("rptr_ty=%s", ty_to_str(tcx, rptr_ty));
            let r = ty::ty_region(tcx, span, rptr_ty);
            infallibly_mk_subr(rcx, true, span, r, bound);
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

    fn guarantor(rcx: @mut Rcx, expr: @ast::expr) -> Option<ty::Region> {
        /*!
         *
         * Computes the guarantor of `expr`, or None if `expr` is
         * not guaranteed by any region.  Here `expr` is some expression
         * whose address is being taken (e.g., there is an expression
         * `&expr`).
         */

        debug!("guarantor(expr=%s)", rcx.fcx.expr_to_str(expr));
        match expr.node {
            ast::expr_unary(ast::deref, b) => {
                let cat = categorize(rcx, b);
                guarantor_of_deref(&cat)
            }
            ast::expr_field(b, _, _) => {
                categorize(rcx, b).guarantor
            }
            ast::expr_index(b, _) => {
                let cat = categorize(rcx, b);
                guarantor_of_deref(&cat)
            }

            ast::expr_paren(e) => {
                guarantor(rcx, e)
            }

            ast::expr_path(*) => {
                // Either a variable or constant and hence resides
                // in constant memory or on the stack frame.  Either way,
                // not guaranteed by a region pointer.
                None
            }

            // All of these expressions are rvalues and hence their
            // value is not guaranteed by a region pointer.
            ast::expr_inline_asm(*) |
            ast::expr_mac(*) |
            ast::expr_lit(_) |
            ast::expr_unary(*) |
            ast::expr_addr_of(*) |
            ast::expr_binary(*) |
            ast::expr_vstore(*) |
            ast::expr_break(*) |
            ast::expr_again(*) |
            ast::expr_ret(*) |
            ast::expr_log(*) |
            ast::expr_while(*) |
            ast::expr_loop(*) |
            ast::expr_assign(*) |
            ast::expr_swap(*) |
            ast::expr_assign_op(*) |
            ast::expr_cast(*) |
            ast::expr_call(*) |
            ast::expr_method_call(*) |
            ast::expr_struct(*) |
            ast::expr_tup(*) |
            ast::expr_if(*) |
            ast::expr_match(*) |
            ast::expr_fn_block(*) |
            ast::expr_loop_body(*) |
            ast::expr_do_body(*) |
            ast::expr_block(*) |
            ast::expr_copy(*) |
            ast::expr_repeat(*) |
            ast::expr_vec(*) => {
                assert!(!ty::expr_is_lval(
                    rcx.fcx.tcx(), rcx.fcx.inh.method_map, expr));
                None
            }
        }
    }

    fn categorize(rcx: @mut Rcx, expr: @ast::expr) -> ExprCategorization {
        debug!("categorize(expr=%s)", rcx.fcx.expr_to_str(expr));
        let _i = ::util::common::indenter();

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

                for adjustment.autoref.each |autoref| {
                    // If there is an autoref, then the result of this
                    // expression will be some sort of borrowed pointer.
                    expr_ct.cat.guarantor = None;
                    expr_ct.cat.pointer = BorrowedPointer(autoref.region);
                    debug!("autoref, cat=%?", expr_ct.cat);
                }
            }

            None => {}
        }

        debug!("result=%?", expr_ct.cat);
        return expr_ct.cat;
    }

    fn categorize_unadjusted(rcx: @mut Rcx,
                             expr: @ast::expr)
                          -> ExprCategorizationType {
        debug!("categorize_unadjusted(expr=%s)", rcx.fcx.expr_to_str(expr));
        let _i = ::util::common::indenter();

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
        expr: @ast::expr,
        autoderefs: uint,
        ct: ExprCategorizationType)
     -> ExprCategorizationType {
        let mut ct = ct;
        let tcx = rcx.fcx.ccx.tcx;
        for uint::range(0, autoderefs) |_| {
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
            ty::ty_estr(ty::vstore_slice(r)) => {
                BorrowedPointer(r)
            }
            ty::ty_uniq(*) |
            ty::ty_estr(ty::vstore_uniq) |
            ty::ty_evec(_, ty::vstore_uniq) => {
                OwnedPointer
            }
            ty::ty_box(*) |
            ty::ty_ptr(*) |
            ty::ty_evec(_, ty::vstore_box) |
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
        pat: @ast::pat,
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
        let _i = ::util::common::indenter();

        match pat.node {
            ast::pat_wild => {}
            ast::pat_ident(ast::bind_by_ref(_), _, opt_p) => {
                link(rcx, pat.span, pat.id, guarantor);

                for opt_p.each |p| {
                    link_ref_bindings_in_pat(rcx, *p, guarantor);
                }
            }
            ast::pat_ident(_, _, opt_p) => {
                for opt_p.each |p| {
                    link_ref_bindings_in_pat(rcx, *p, guarantor);
                }
            }
            ast::pat_enum(*) => {}
            ast::pat_struct(_, ref fpats, _) => {
                for fpats.each |fpat| {
                    link_ref_bindings_in_pat(rcx, fpat.pat, guarantor);
                }
            }
            ast::pat_tup(ref ps) => {
                link_ref_bindings_in_pats(rcx, ps, guarantor)
            }
            ast::pat_box(p) => {
                link_ref_bindings_in_pat(rcx, p, None)
            }
            ast::pat_uniq(p) => {
                link_ref_bindings_in_pat(rcx, p, guarantor)
            }
            ast::pat_region(p) => {
                let rptr_ty = rcx.resolve_node_type(pat.id);
                if !ty::type_is_error(rptr_ty) {
                    let r = ty::ty_region(rcx.fcx.tcx(), pat.span, rptr_ty);
                    link_ref_bindings_in_pat(rcx, p, Some(r));
                }
            }
            ast::pat_lit(*) => {}
            ast::pat_range(*) => {}
            ast::pat_vec(ref before, ref slice, ref after) => {
                let vec_ty = rcx.resolve_node_type(pat.id);
                if !ty::type_is_error(vec_ty) {
                    let vstore = ty::ty_vstore(vec_ty);
                    let guarantor1 = match vstore {
                        ty::vstore_fixed(_) | ty::vstore_uniq => guarantor,
                        ty::vstore_slice(r) => Some(r),
                        ty::vstore_box => None
                    };

                    link_ref_bindings_in_pats(rcx, before, guarantor1);
                    for slice.each |&p| {
                        link_ref_bindings_in_pat(rcx, p, guarantor);
                    }
                    link_ref_bindings_in_pats(rcx, after, guarantor1);
                }
            }
        }
    }

    fn link_ref_bindings_in_pats(rcx: @mut Rcx,
                                 pats: &~[@ast::pat],
                                 guarantor: Option<ty::Region>) {
        for pats.each |pat| {
            link_ref_bindings_in_pat(rcx, *pat, guarantor);
        }
    }

}

pub fn infallibly_mk_subr(rcx: @mut Rcx,
                          a_is_expected: bool,
                          span: span,
                          a: ty::Region,
                          b: ty::Region) {
    /*!
     * Constrains `a` to be a subregion of `b`.  In many cases, we
     * know that this can never yield an error due to the way that
     * region inferencing works.  Therefore just report a bug if we
     * ever see `Err(_)`.
     */

    match rcx.fcx.mk_subr(a_is_expected, span, a, b) {
        result::Ok(()) => {}
        result::Err(e) => {
            rcx.fcx.ccx.tcx.sess.span_bug(
                span,
                fmt!("Supposedly infallible attempt to \
                      make %? < %? failed: %?",
                     a, b, e));
        }
    }
}
