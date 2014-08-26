// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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

use middle::def;
use middle::pat_util;
use middle::ty;
use middle::ty_fold::{TypeFolder,TypeFoldable};
use middle::typeck::astconv::AstConv;
use middle::typeck::check::FnCtxt;
use middle::typeck::infer::{force_all, resolve_all, resolve_region};
use middle::typeck::infer::resolve_type;
use middle::typeck::infer;
use middle::typeck::{MethodCall, MethodCallee};
use middle::typeck::vtable_res;
use middle::typeck::write_substs_to_tcx;
use middle::typeck::write_ty_to_tcx;
use util::ppaux::Repr;

use std::cell::Cell;

use syntax::ast;
use syntax::codemap::{DUMMY_SP, Span};
use syntax::print::pprust::pat_to_string;
use syntax::visit;
use syntax::visit::Visitor;

///////////////////////////////////////////////////////////////////////////
// Entry point functions

pub fn resolve_type_vars_in_expr(fcx: &FnCtxt, e: &ast::Expr) {
    assert_eq!(fcx.writeback_errors.get(), false);
    let mut wbcx = WritebackCx::new(fcx);
    wbcx.visit_expr(e, ());
    wbcx.visit_upvar_borrow_map();
    wbcx.visit_unboxed_closures();
}

pub fn resolve_type_vars_in_fn(fcx: &FnCtxt,
                               decl: &ast::FnDecl,
                               blk: &ast::Block) {
    assert_eq!(fcx.writeback_errors.get(), false);
    let mut wbcx = WritebackCx::new(fcx);
    wbcx.visit_block(blk, ());
    for arg in decl.inputs.iter() {
        wbcx.visit_pat(&*arg.pat, ());

        // Privacy needs the type for the whole pattern, not just each binding
        if !pat_util::pat_is_binding(&fcx.tcx().def_map, &*arg.pat) {
            wbcx.visit_node_id(ResolvingPattern(arg.pat.span),
                               arg.pat.id);
        }
    }
    wbcx.visit_upvar_borrow_map();
    wbcx.visit_unboxed_closures();
}

pub fn resolve_impl_res(infcx: &infer::InferCtxt,
                        span: Span,
                        vtable_res: &vtable_res)
                        -> vtable_res {
    let errors = Cell::new(false); // nobody cares
    let mut resolver = Resolver::from_infcx(infcx,
                                            &errors,
                                            ResolvingImplRes(span));
    vtable_res.resolve_in(&mut resolver)
}

///////////////////////////////////////////////////////////////////////////
// The Writerback context. This visitor walks the AST, checking the
// fn-specific tables to find references to types or regions. It
// resolves those regions to remove inference variables and writes the
// final result back into the master tables in the tcx. Here and
// there, it applies a few ad-hoc checks that were not convenient to
// do elsewhere.

struct WritebackCx<'cx> {
    fcx: &'cx FnCtxt<'cx>,
}

impl<'cx> WritebackCx<'cx> {
    fn new(fcx: &'cx FnCtxt) -> WritebackCx<'cx> {
        WritebackCx { fcx: fcx }
    }

    fn tcx(&self) -> &'cx ty::ctxt {
        self.fcx.tcx()
    }
}

///////////////////////////////////////////////////////////////////////////
// Impl of Visitor for Resolver
//
// This is the master code which walks the AST. It delegates most of
// the heavy lifting to the generic visit and resolve functions
// below. In general, a function is made into a `visitor` if it must
// traffic in node-ids or update tables in the type context etc.

impl<'cx> Visitor<()> for WritebackCx<'cx> {
    fn visit_item(&mut self, _: &ast::Item, _: ()) {
        // Ignore items
    }

    fn visit_stmt(&mut self, s: &ast::Stmt, _: ()) {
        if self.fcx.writeback_errors.get() {
            return;
        }

        self.visit_node_id(ResolvingExpr(s.span), ty::stmt_node_id(s));
        visit::walk_stmt(self, s, ());
    }

    fn visit_expr(&mut self, e:&ast::Expr, _: ()) {
        if self.fcx.writeback_errors.get() {
            return;
        }

        self.visit_node_id(ResolvingExpr(e.span), e.id);
        self.visit_method_map_entry(ResolvingExpr(e.span),
                                    MethodCall::expr(e.id));
        self.visit_vtable_map_entry(ResolvingExpr(e.span),
                                    MethodCall::expr(e.id));

        match e.node {
            ast::ExprFnBlock(_, ref decl, _) |
            ast::ExprProc(ref decl, _) |
            ast::ExprUnboxedFn(_, _, ref decl, _) => {
                for input in decl.inputs.iter() {
                    let _ = self.visit_node_id(ResolvingExpr(e.span),
                                               input.id);
                }
            }
            _ => {}
        }

        visit::walk_expr(self, e, ());
    }

    fn visit_block(&mut self, b: &ast::Block, _: ()) {
        if self.fcx.writeback_errors.get() {
            return;
        }

        self.visit_node_id(ResolvingExpr(b.span), b.id);
        visit::walk_block(self, b, ());
    }

    fn visit_pat(&mut self, p: &ast::Pat, _: ()) {
        if self.fcx.writeback_errors.get() {
            return;
        }

        self.visit_node_id(ResolvingPattern(p.span), p.id);

        debug!("Type for pattern binding {} (id {}) resolved to {}",
               pat_to_string(p),
               p.id,
               ty::node_id_to_type(self.tcx(), p.id).repr(self.tcx()));

        visit::walk_pat(self, p, ());
    }

    fn visit_local(&mut self, l: &ast::Local, _: ()) {
        if self.fcx.writeback_errors.get() {
            return;
        }

        let var_ty = self.fcx.local_ty(l.span, l.id);
        let var_ty = self.resolve(&var_ty, ResolvingLocal(l.span));
        write_ty_to_tcx(self.tcx(), l.id, var_ty);
        visit::walk_local(self, l, ());
    }

    fn visit_ty(&mut self, t: &ast::Ty, _: ()) {
        match t.node {
            ast::TyFixedLengthVec(ref ty, ref count_expr) => {
                self.visit_ty(&**ty, ());
                write_ty_to_tcx(self.tcx(), count_expr.id, ty::mk_uint());
            }
            _ => visit::walk_ty(self, t, ())
        }
    }
}

impl<'cx> WritebackCx<'cx> {
    fn visit_upvar_borrow_map(&self) {
        if self.fcx.writeback_errors.get() {
            return;
        }

        for (upvar_id, upvar_borrow) in self.fcx.inh.upvar_borrow_map.borrow().iter() {
            let r = upvar_borrow.region;
            let r = self.resolve(&r, ResolvingUpvar(*upvar_id));
            let new_upvar_borrow = ty::UpvarBorrow { kind: upvar_borrow.kind,
                                                     region: r };
            debug!("Upvar borrow for {} resolved to {}",
                   upvar_id.repr(self.tcx()),
                   new_upvar_borrow.repr(self.tcx()));
            self.fcx.tcx().upvar_borrow_map.borrow_mut().insert(
                *upvar_id, new_upvar_borrow);
        }
    }

    fn visit_unboxed_closures(&self) {
        if self.fcx.writeback_errors.get() {
            return
        }

        for (def_id, unboxed_closure) in self.fcx
                                             .inh
                                             .unboxed_closures
                                             .borrow()
                                             .iter() {
            let closure_ty = self.resolve(&unboxed_closure.closure_type,
                                          ResolvingUnboxedClosure(*def_id));
            let unboxed_closure = ty::UnboxedClosure {
                closure_type: closure_ty,
                kind: unboxed_closure.kind,
            };
            self.fcx
                .tcx()
                .unboxed_closures
                .borrow_mut()
                .insert(*def_id, unboxed_closure);
        }
    }

    fn visit_node_id(&self, reason: ResolveReason, id: ast::NodeId) {
        // Resolve any borrowings for the node with id `id`
        self.visit_adjustments(reason, id);

        // Resolve the type of the node with id `id`
        let n_ty = self.fcx.node_ty(id);
        let n_ty = self.resolve(&n_ty, reason);
        write_ty_to_tcx(self.tcx(), id, n_ty);
        debug!("Node {} has type {}", id, n_ty.repr(self.tcx()));

        // Resolve any substitutions
        self.fcx.opt_node_ty_substs(id, |item_substs| {
            write_substs_to_tcx(self.tcx(), id,
                                self.resolve(item_substs, reason));
        });
    }

    fn visit_adjustments(&self, reason: ResolveReason, id: ast::NodeId) {
        match self.fcx.inh.adjustments.borrow_mut().pop(&id) {
            None => {
                debug!("No adjustments for node {}", id);
            }

            Some(adjustment) => {
                let adj_object = ty::adjust_is_object(&adjustment);
                let resolved_adjustment = match adjustment {
                    ty::AutoAddEnv(store) => {
                        // FIXME(eddyb) #2190 Allow only statically resolved
                        // bare functions to coerce to a closure to avoid
                        // constructing (slower) indirect call wrappers.
                        match self.tcx().def_map.borrow().find(&id) {
                            Some(&def::DefFn(..)) |
                            Some(&def::DefStaticMethod(..)) |
                            Some(&def::DefVariant(..)) |
                            Some(&def::DefStruct(_)) => {
                            }
                            _ => {
                                span_err!(self.tcx().sess, reason.span(self.tcx()), E0100,
                                    "cannot coerce non-statically resolved bare fn");
                            }
                        }

                        ty::AutoAddEnv(self.resolve(&store, reason))
                    }

                    ty::AutoDerefRef(adj) => {
                        for autoderef in range(0, adj.autoderefs) {
                            let method_call = MethodCall::autoderef(id, autoderef);
                            self.visit_method_map_entry(reason, method_call);
                            self.visit_vtable_map_entry(reason, method_call);
                        }

                        if adj_object {
                            let method_call = MethodCall::autoobject(id);
                            self.visit_method_map_entry(reason, method_call);
                            self.visit_vtable_map_entry(reason, method_call);
                        }

                        ty::AutoDerefRef(ty::AutoDerefRef {
                            autoderefs: adj.autoderefs,
                            autoref: self.resolve(&adj.autoref, reason),
                        })
                    }
                };
                debug!("Adjustments for node {}: {:?}", id, resolved_adjustment);
                self.tcx().adjustments.borrow_mut().insert(
                    id, resolved_adjustment);
            }
        }
    }

    fn visit_method_map_entry(&self,
                              reason: ResolveReason,
                              method_call: MethodCall) {
        // Resolve any method map entry
        match self.fcx.inh.method_map.borrow_mut().pop(&method_call) {
            Some(method) => {
                debug!("writeback::resolve_method_map_entry(call={:?}, entry={})",
                       method_call,
                       method.repr(self.tcx()));
                let new_method = MethodCallee {
                    origin: method.origin,
                    ty: self.resolve(&method.ty, reason),
                    substs: self.resolve(&method.substs, reason),
                };

                self.tcx().method_map.borrow_mut().insert(
                    method_call,
                    new_method);
            }
            None => {}
        }
    }

    fn visit_vtable_map_entry(&self,
                              reason: ResolveReason,
                              vtable_key: MethodCall) {
        // Resolve any vtable map entry
        match self.fcx.inh.vtable_map.borrow_mut().pop(&vtable_key) {
            Some(origins) => {
                let r_origins = self.resolve(&origins, reason);
                debug!("writeback::resolve_vtable_map_entry(\
                        vtable_key={}, vtables={:?})",
                       vtable_key, r_origins.repr(self.tcx()));
                self.tcx().vtable_map.borrow_mut().insert(vtable_key, r_origins);
            }
            None => {}
        }
    }

    fn resolve<T:ResolveIn>(&self, t: &T, reason: ResolveReason) -> T {
        t.resolve_in(&mut Resolver::new(self.fcx, reason))
    }
}

///////////////////////////////////////////////////////////////////////////
// Resolution reason.

enum ResolveReason {
    ResolvingExpr(Span),
    ResolvingLocal(Span),
    ResolvingPattern(Span),
    ResolvingUpvar(ty::UpvarId),
    ResolvingImplRes(Span),
    ResolvingUnboxedClosure(ast::DefId),
}

impl ResolveReason {
    fn span(&self, tcx: &ty::ctxt) -> Span {
        match *self {
            ResolvingExpr(s) => s,
            ResolvingLocal(s) => s,
            ResolvingPattern(s) => s,
            ResolvingUpvar(upvar_id) => {
                ty::expr_span(tcx, upvar_id.closure_expr_id)
            }
            ResolvingImplRes(s) => s,
            ResolvingUnboxedClosure(did) => {
                if did.krate == ast::LOCAL_CRATE {
                    ty::expr_span(tcx, did.node)
                } else {
                    DUMMY_SP
                }
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Convenience methods for resolving different kinds of things.

trait ResolveIn {
    fn resolve_in(&self, resolver: &mut Resolver) -> Self;
}

impl<T:TypeFoldable> ResolveIn for T {
    fn resolve_in(&self, resolver: &mut Resolver) -> T {
        self.fold_with(resolver)
    }
}

///////////////////////////////////////////////////////////////////////////
// The Resolver. This is the type folding engine that detects
// unresolved types and so forth.

struct Resolver<'cx> {
    tcx: &'cx ty::ctxt,
    infcx: &'cx infer::InferCtxt<'cx>,
    writeback_errors: &'cx Cell<bool>,
    reason: ResolveReason,
}

impl<'cx> Resolver<'cx> {
    fn new(fcx: &'cx FnCtxt<'cx>,
           reason: ResolveReason)
           -> Resolver<'cx>
    {
        Resolver { infcx: fcx.infcx(),
                   tcx: fcx.tcx(),
                   writeback_errors: &fcx.writeback_errors,
                   reason: reason }
    }

    fn from_infcx(infcx: &'cx infer::InferCtxt<'cx>,
                  writeback_errors: &'cx Cell<bool>,
                  reason: ResolveReason)
                  -> Resolver<'cx>
    {
        Resolver { infcx: infcx,
                   tcx: infcx.tcx,
                   writeback_errors: writeback_errors,
                   reason: reason }
    }

    fn report_error(&self, e: infer::fixup_err) {
        self.writeback_errors.set(true);
        if !self.tcx.sess.has_errors() {
            match self.reason {
                ResolvingExpr(span) => {
                    span_err!(self.tcx.sess, span, E0101,
                        "cannot determine a type for this expression: {}",
                        infer::fixup_err_to_string(e));
                }

                ResolvingLocal(span) => {
                    span_err!(self.tcx.sess, span, E0102,
                        "cannot determine a type for this local variable: {}",
                        infer::fixup_err_to_string(e));
                }

                ResolvingPattern(span) => {
                    span_err!(self.tcx.sess, span, E0103,
                        "cannot determine a type for this pattern binding: {}",
                        infer::fixup_err_to_string(e));
                }

                ResolvingUpvar(upvar_id) => {
                    let span = self.reason.span(self.tcx);
                    span_err!(self.tcx.sess, span, E0104,
                        "cannot resolve lifetime for captured variable `{}`: {}",
                        ty::local_var_name_str(self.tcx, upvar_id.var_id).get().to_string(),
                        infer::fixup_err_to_string(e));
                }

                ResolvingImplRes(span) => {
                    span_err!(self.tcx.sess, span, E0105,
                        "cannot determine a type for impl supertrait");
                }

                ResolvingUnboxedClosure(_) => {
                    let span = self.reason.span(self.tcx);
                    self.tcx.sess.span_err(span,
                                           "cannot determine a type for this \
                                            unboxed closure")
                }
            }
        }
    }
}

impl<'cx> TypeFolder for Resolver<'cx> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt {
        self.tcx
    }

    fn fold_ty(&mut self, t: ty::t) -> ty::t {
        if !ty::type_needs_infer(t) {
            return t;
        }

        match resolve_type(self.infcx, None, t, resolve_all | force_all) {
            Ok(t) => t,
            Err(e) => {
                self.report_error(e);
                ty::mk_err()
            }
        }
    }

    fn fold_region(&mut self, r: ty::Region) -> ty::Region {
        match resolve_region(self.infcx, r, resolve_all | force_all) {
            Ok(r) => r,
            Err(e) => {
                self.report_error(e);
                ty::ReStatic
            }
        }
    }
}
