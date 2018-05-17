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

use super::{FnCtxt, PlaceOp, Needs};
use super::method::MethodCallee;

use rustc::infer::InferOk;
use rustc::session::DiagnosticMessageId;
use rustc::traits;
use rustc::ty::{self, Ty, TraitRef};
use rustc::ty::{ToPredicate, TypeFoldable};
use rustc::ty::adjustment::{Adjustment, Adjust, OverloadedDeref};

use syntax_pos::Span;
use syntax::symbol::Symbol;

use std::iter;

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
    include_raw_pointers: bool,
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

        if self.steps.len() >= *tcx.sess.recursion_limit.get() {
            // We've reached the recursion limit, error gracefully.
            let suggested_limit = *tcx.sess.recursion_limit.get() * 2;
            let msg = format!("reached the recursion limit while auto-dereferencing {:?}",
                              self.cur_ty);
            let error_id = (DiagnosticMessageId::ErrorId(55), Some(self.span), msg.clone());
            let fresh = tcx.sess.one_time_diagnostics.borrow_mut().insert(error_id);
            if fresh {
                struct_span_err!(tcx.sess,
                                 self.span,
                                 E0055,
                                 "reached the recursion limit while auto-dereferencing {:?}",
                                 self.cur_ty)
                    .span_label(self.span, "deref recursion limit reached")
                    .help(&format!(
                        "consider adding a `#![recursion_limit=\"{}\"]` attribute to your crate",
                        suggested_limit))
                    .emit();
            }
            return None;
        }

        if self.cur_ty.is_ty_var() {
            return None;
        }

        // Otherwise, deref if type is derefable:
        let (kind, new_ty) =
            if let Some(mt) = self.cur_ty.builtin_deref(self.include_raw_pointers) {
                (AutoderefKind::Builtin, mt.ty)
            } else {
                let ty = self.overloaded_deref_ty(self.cur_ty)?;
                (AutoderefKind::Overloaded, ty)
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
            def_id: tcx.lang_items().deref_trait()?,
            substs: tcx.mk_substs_trait(self.cur_ty, &[]),
        };

        let cause = traits::ObligationCause::misc(self.span, self.fcx.body_id);

        let obligation = traits::Obligation::new(cause.clone(),
                                                 self.fcx.param_env,
                                                 trait_ref.to_predicate());
        if !self.fcx.predicate_may_hold(&obligation) {
            debug!("overloaded_deref_ty: cannot match obligation");
            return None;
        }

        let mut selcx = traits::SelectionContext::new(self.fcx);
        let normalized_ty = traits::normalize_projection_type(&mut selcx,
                                                              self.fcx.param_env,
                                                              ty::ProjectionTy::from_ref_and_name(
                                                                  tcx,
                                                                  trait_ref,
                                                                  Symbol::intern("Target"),
                                                              ),
                                                              cause,
                                                              0,
                                                              &mut self.obligations);

        debug!("overloaded_deref_ty({:?}) = {:?}", ty, normalized_ty);

        Some(self.fcx.resolve_type_vars_if_possible(&normalized_ty))
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

    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Returns the adjustment steps.
    pub fn adjust_steps(&self, needs: Needs)
                        -> Vec<Adjustment<'tcx>> {
        self.fcx.register_infer_ok_obligations(self.adjust_steps_as_infer_ok(needs))
    }

    pub fn adjust_steps_as_infer_ok(&self, needs: Needs)
                                    -> InferOk<'tcx, Vec<Adjustment<'tcx>>> {
        let mut obligations = vec![];
        let targets = self.steps.iter().skip(1).map(|&(ty, _)| ty)
            .chain(iter::once(self.cur_ty));
        let steps: Vec<_> = self.steps.iter().map(|&(source, kind)| {
            if let AutoderefKind::Overloaded = kind {
                self.fcx.try_overloaded_deref(self.span, source, needs)
                    .and_then(|InferOk { value: method, obligations: o }| {
                        obligations.extend(o);
                        if let ty::TyRef(region, _, mutbl) = method.sig.output().sty {
                            Some(OverloadedDeref {
                                region,
                                mutbl,
                            })
                        } else {
                            None
                        }
                    })
            } else {
                None
            }
        }).zip(targets).map(|(autoderef, target)| {
            Adjustment {
                kind: Adjust::Deref(autoderef),
                target
            }
        }).collect();

        InferOk {
            obligations,
            value: steps
        }
    }

    /// also dereference through raw pointer types
    /// e.g. assuming ptr_to_Foo is the type `*const Foo`
    /// fcx.autoderef(span, ptr_to_Foo)  => [*const Foo]
    /// fcx.autoderef(span, ptr_to_Foo).include_raw_ptrs() => [*const Foo, Foo]
    pub fn include_raw_pointers(mut self) -> Self {
        self.include_raw_pointers = true;
        self
    }

    pub fn finalize(self) {
        let fcx = self.fcx;
        fcx.register_predicates(self.into_obligations());
    }

    pub fn into_obligations(self) -> Vec<traits::PredicateObligation<'tcx>> {
        self.obligations
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
            include_raw_pointers: false,
            span,
        }
    }

    pub fn try_overloaded_deref(&self,
                                span: Span,
                                base_ty: Ty<'tcx>,
                                needs: Needs)
                                -> Option<InferOk<'tcx, MethodCallee<'tcx>>> {
        self.try_overloaded_place_op(span, base_ty, &[], needs, PlaceOp::Deref)
    }
}
