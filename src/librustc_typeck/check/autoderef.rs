// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use astconv::AstConv;

use super::{FnCtxt, LvalueOp};

use check::coercion::AsCoercionSite;
use rustc::infer::InferOk;
use rustc::traits;
use rustc::ty::{self, Ty, TraitRef};
use rustc::ty::{ToPredicate, TypeFoldable};
use rustc::ty::{MethodCall, MethodCallee};
use rustc::ty::{LvaluePreference, NoPreference};
use rustc::hir;

use syntax_pos::Span;
use syntax::symbol::Symbol;

#[derive(Copy, Clone, Debug)]
enum AutoderefKind {
    Builtin,
    Overloaded,
}

pub struct Autoderef<'a, 'gcx: 'tcx, 'tcx: 'a> {
    fcx: &'a FnCtxt<'a, 'gcx, 'tcx>,
    steps: Vec<(Ty<'tcx>, AutoderefKind)>,
    cur_ty: Ty<'tcx>,
    obligations: Vec<traits::PredicateObligation<'tcx>>,
    at_start: bool,
    span: Span,
}

impl<'a, 'gcx, 'tcx> Iterator for Autoderef<'a, 'gcx, 'tcx> {
    type Item = (Ty<'tcx>, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let tcx = self.fcx.tcx;

        debug!("autoderef: steps={:?}, cur_ty={:?}",
               self.steps,
               self.cur_ty);
        if self.at_start {
            self.at_start = false;
            debug!("autoderef stage #0 is {:?}", self.cur_ty);
            return Some((self.cur_ty, 0));
        }

        if self.steps.len() == tcx.sess.recursion_limit.get() {
            // We've reached the recursion limit, error gracefully.
            let suggested_limit = tcx.sess.recursion_limit.get() * 2;
            struct_span_err!(tcx.sess,
                             self.span,
                             E0055,
                             "reached the recursion limit while auto-dereferencing {:?}",
                             self.cur_ty)
                .span_label(self.span, "deref recursion limit reached")
                .help(&format!(
                        "consider adding a `#[recursion_limit=\"{}\"]` attribute to your crate",
                        suggested_limit))
                .emit();
            return None;
        }

        if self.cur_ty.is_ty_var() {
            return None;
        }

        // Otherwise, deref if type is derefable:
        let (kind, new_ty) = if let Some(mt) = self.cur_ty.builtin_deref(false, NoPreference) {
            (AutoderefKind::Builtin, mt.ty)
        } else {
            match self.overloaded_deref_ty(self.cur_ty) {
                Some(ty) => (AutoderefKind::Overloaded, ty),
                _ => return None,
            }
        };

        if new_ty.references_error() {
            return None;
        }

        self.steps.push((self.cur_ty, kind));
        debug!("autoderef stage #{:?} is {:?} from {:?}",
               self.steps.len(),
               new_ty,
               (self.cur_ty, kind));
        self.cur_ty = new_ty;

        Some((self.cur_ty, self.steps.len()))
    }
}

impl<'a, 'gcx, 'tcx> Autoderef<'a, 'gcx, 'tcx> {
    fn overloaded_deref_ty(&mut self, ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
        debug!("overloaded_deref_ty({:?})", ty);

        let tcx = self.fcx.tcx();

        // <cur_ty as Deref>
        let trait_ref = TraitRef {
            def_id: match tcx.lang_items.deref_trait() {
                Some(f) => f,
                None => return None,
            },
            substs: tcx.mk_substs_trait(self.cur_ty, &[]),
        };

        let cause = traits::ObligationCause::misc(self.span, self.fcx.body_id);

        let mut selcx = traits::SelectionContext::new(self.fcx);
        let obligation = traits::Obligation::new(cause.clone(), trait_ref.to_predicate());
        if !selcx.evaluate_obligation(&obligation) {
            debug!("overloaded_deref_ty: cannot match obligation");
            return None;
        }

        let normalized = traits::normalize_projection_type(&mut selcx,
                                                           ty::ProjectionTy {
                                                               trait_ref: trait_ref,
                                                               item_name: Symbol::intern("Target"),
                                                           },
                                                           cause,
                                                           0);

        debug!("overloaded_deref_ty({:?}) = {:?}", ty, normalized);
        self.obligations.extend(normalized.obligations);

        Some(self.fcx.resolve_type_vars_if_possible(&normalized.value))
    }

    /// Returns the final type, generating an error if it is an
    /// unresolved inference variable.
    pub fn unambiguous_final_ty(&self) -> Ty<'tcx> {
        self.fcx.structurally_resolved_type(self.span, self.cur_ty)
    }

    /// Returns the final type we ended up with, which may well be an
    /// inference variable (we will resolve it first, if possible).
    pub fn maybe_ambiguous_final_ty(&self) -> Ty<'tcx> {
        self.fcx.resolve_type_vars_if_possible(&self.cur_ty)
    }

    pub fn finalize(self, pref: LvaluePreference, expr: &hir::Expr) {
        let fcx = self.fcx;
        fcx.register_infer_ok_obligations(self.finalize_as_infer_ok(pref, &[expr]));
    }

    pub fn finalize_as_infer_ok<E>(self, pref: LvaluePreference, exprs: &[E])
                                   -> InferOk<'tcx, ()>
        where E: AsCoercionSite
    {
        let Autoderef { fcx, span, mut obligations, steps, .. } = self;
        let methods: Vec<_> = steps
            .iter()
            .map(|&(ty, kind)| {
                if let AutoderefKind::Overloaded = kind {
                    fcx.try_overloaded_deref(span, None, ty, pref)
                        .map(|InferOk { value, obligations: o }| {
                            obligations.extend(o);
                            value
                        })
                } else {
                    None
                }
            })
            .collect();

        debug!("finalize({:?}) - {:?},{:?}",
               pref,
               methods,
               obligations);

        for expr in exprs {
            let expr = expr.as_coercion_site();
            debug!("finalize - finalizing #{} - {:?}", expr.id, expr);
            for (n, method) in methods.iter().enumerate() {
                if let &Some(method) = method {
                    let method_call = MethodCall::autoderef(expr.id, n as u32);
                    fcx.tables.borrow_mut().method_map.insert(method_call, method);
                }
            }
        }

        InferOk {
            value: (),
            obligations
        }
    }
}

impl<'a, 'gcx, 'tcx> FnCtxt<'a, 'gcx, 'tcx> {
    pub fn autoderef(&'a self, span: Span, base_ty: Ty<'tcx>) -> Autoderef<'a, 'gcx, 'tcx> {
        Autoderef {
            fcx: self,
            steps: vec![],
            cur_ty: self.resolve_type_vars_if_possible(&base_ty),
            obligations: vec![],
            at_start: true,
            span: span,
        }
    }

    pub fn try_overloaded_deref(&self,
                                span: Span,
                                base_expr: Option<&hir::Expr>,
                                base_ty: Ty<'tcx>,
                                pref: LvaluePreference)
                                -> Option<InferOk<'tcx, MethodCallee<'tcx>>> {
        let rcvr = base_expr.map(|base_expr| super::AdjustedRcvr {
            rcvr_expr: base_expr, autoderefs: 0, unsize: false
        });

        self.try_overloaded_lvalue_op(span, rcvr, base_ty, &[], pref, LvalueOp::Deref)
    }
}
