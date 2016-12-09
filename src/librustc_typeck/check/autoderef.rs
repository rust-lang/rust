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

use super::FnCtxt;

use rustc::traits;
use rustc::ty::{self, Ty, TraitRef};
use rustc::ty::{ToPredicate, TypeFoldable};
use rustc::ty::{MethodCall, MethodCallee};
use rustc::ty::{LvaluePreference, NoPreference, PreferMutLvalue};
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
            struct_span_err!(tcx.sess,
                             self.span,
                             E0055,
                             "reached the recursion limit while auto-dereferencing {:?}",
                             self.cur_ty)
                .span_label(self.span, &format!("deref recursion limit reached"))
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

    pub fn finalize<'b, I>(self, pref: LvaluePreference, exprs: I)
        where I: IntoIterator<Item = &'b hir::Expr>
    {
        let methods: Vec<_> = self.steps
            .iter()
            .map(|&(ty, kind)| {
                if let AutoderefKind::Overloaded = kind {
                    self.fcx.try_overloaded_deref(self.span, None, ty, pref)
                } else {
                    None
                }
            })
            .collect();

        debug!("finalize({:?}) - {:?},{:?}",
               pref,
               methods,
               self.obligations);

        for expr in exprs {
            debug!("finalize - finalizing #{} - {:?}", expr.id, expr);
            for (n, method) in methods.iter().enumerate() {
                if let &Some(method) = method {
                    let method_call = MethodCall::autoderef(expr.id, n as u32);
                    self.fcx.tables.borrow_mut().method_map.insert(method_call, method);
                }
            }
        }

        for obligation in self.obligations {
            self.fcx.register_predicate(obligation);
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
                                lvalue_pref: LvaluePreference)
                                -> Option<MethodCallee<'tcx>> {
        debug!("try_overloaded_deref({:?},{:?},{:?},{:?})",
               span,
               base_expr,
               base_ty,
               lvalue_pref);
        // Try DerefMut first, if preferred.
        let method = match (lvalue_pref, self.tcx.lang_items.deref_mut_trait()) {
            (PreferMutLvalue, Some(trait_did)) => {
                self.lookup_method_in_trait(span,
                                            base_expr,
                                            Symbol::intern("deref_mut"),
                                            trait_did,
                                            base_ty,
                                            None)
            }
            _ => None,
        };

        // Otherwise, fall back to Deref.
        let method = match (method, self.tcx.lang_items.deref_trait()) {
            (None, Some(trait_did)) => {
                self.lookup_method_in_trait(span,
                                            base_expr,
                                            Symbol::intern("deref"),
                                            trait_did,
                                            base_ty,
                                            None)
            }
            (method, _) => method,
        };

        method
    }
}
