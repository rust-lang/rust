use super::{FnCtxt, PlaceOp, Needs};
use super::method::MethodCallee;

use rustc::hir;
use rustc::infer::{InferCtxt, InferOk};
use rustc::session::DiagnosticMessageId;
use rustc::traits::{self, TraitEngine};
use rustc::ty::{self, Ty, TyCtxt, TraitRef};
use rustc::ty::{ToPredicate, TypeFoldable};
use rustc::ty::adjustment::{Adjustment, Adjust, OverloadedDeref};

use syntax_pos::Span;
use syntax::ast::Ident;

use std::iter;

#[derive(Copy, Clone, Debug)]
enum AutoderefKind {
    Builtin,
    Overloaded,
}

pub struct Autoderef<'a, 'tcx> {
    infcx: &'a InferCtxt<'a, 'tcx>,
    body_id: hir::HirId,
    param_env: ty::ParamEnv<'tcx>,
    steps: Vec<(Ty<'tcx>, AutoderefKind)>,
    cur_ty: Ty<'tcx>,
    obligations: Vec<traits::PredicateObligation<'tcx>>,
    at_start: bool,
    include_raw_pointers: bool,
    span: Span,
    silence_errors: bool,
    reached_recursion_limit: bool,
}

impl<'a, 'tcx> Iterator for Autoderef<'a, 'tcx> {
    type Item = (Ty<'tcx>, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let tcx = self.infcx.tcx;

        debug!("autoderef: steps={:?}, cur_ty={:?}",
               self.steps,
               self.cur_ty);
        if self.at_start {
            self.at_start = false;
            debug!("autoderef stage #0 is {:?}", self.cur_ty);
            return Some((self.cur_ty, 0));
        }

        if self.steps.len() >= *tcx.sess.recursion_limit.get() {
            if !self.silence_errors {
                report_autoderef_recursion_limit_error(tcx, self.span, self.cur_ty);
            }
            self.reached_recursion_limit = true;
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

impl<'a, 'tcx> Autoderef<'a, 'tcx> {
    pub fn new(
        infcx: &'a InferCtxt<'a, 'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        body_id: hir::HirId,
        span: Span,
        base_ty: Ty<'tcx>,
    ) -> Autoderef<'a, 'tcx> {
        Autoderef {
            infcx,
            body_id,
            param_env,
            steps: vec![],
            cur_ty: infcx.resolve_vars_if_possible(&base_ty),
            obligations: vec![],
            at_start: true,
            include_raw_pointers: false,
            silence_errors: false,
            reached_recursion_limit: false,
            span,
        }
    }

    fn overloaded_deref_ty(&mut self, ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
        debug!("overloaded_deref_ty({:?})", ty);

        let tcx = self.infcx.tcx;

        // <cur_ty as Deref>
        let trait_ref = TraitRef {
            def_id: tcx.lang_items().deref_trait()?,
            substs: tcx.mk_substs_trait(self.cur_ty, &[]),
        };

        let cause = traits::ObligationCause::misc(self.span, self.body_id);

        let obligation = traits::Obligation::new(cause.clone(),
                                                 self.param_env,
                                                 trait_ref.to_predicate());
        if !self.infcx.predicate_may_hold(&obligation) {
            debug!("overloaded_deref_ty: cannot match obligation");
            return None;
        }

        let mut fulfillcx = traits::FulfillmentContext::new_in_snapshot();
        let normalized_ty = fulfillcx.normalize_projection_type(
            &self.infcx,
            self.param_env,
            ty::ProjectionTy::from_ref_and_name(
                tcx,
                trait_ref,
                Ident::from_str("Target"),
            ),
            cause);
        if let Err(e) = fulfillcx.select_where_possible(&self.infcx) {
            // This shouldn't happen, except for evaluate/fulfill mismatches,
            // but that's not a reason for an ICE (`predicate_may_hold` is conservative
            // by design).
            debug!("overloaded_deref_ty: encountered errors {:?} while fulfilling",
                   e);
            return None;
        }
        let obligations = fulfillcx.pending_obligations();
        debug!("overloaded_deref_ty({:?}) = ({:?}, {:?})",
               ty, normalized_ty, obligations);
        self.obligations.extend(obligations);

        Some(self.infcx.resolve_vars_if_possible(&normalized_ty))
    }

    /// Returns the final type, generating an error if it is an
    /// unresolved inference variable.
    pub fn unambiguous_final_ty(&self, fcx: &FnCtxt<'a, 'tcx>) -> Ty<'tcx> {
        fcx.structurally_resolved_type(self.span, self.cur_ty)
    }

    /// Returns the final type we ended up with, which may well be an
    /// inference variable (we will resolve it first, if possible).
    pub fn maybe_ambiguous_final_ty(&self) -> Ty<'tcx> {
        self.infcx.resolve_vars_if_possible(&self.cur_ty)
    }

    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Returns the adjustment steps.
    pub fn adjust_steps(&self, fcx: &FnCtxt<'a, 'tcx>, needs: Needs) -> Vec<Adjustment<'tcx>> {
        fcx.register_infer_ok_obligations(self.adjust_steps_as_infer_ok(fcx, needs))
    }

    pub fn adjust_steps_as_infer_ok(
        &self,
        fcx: &FnCtxt<'a, 'tcx>,
        needs: Needs,
    ) -> InferOk<'tcx, Vec<Adjustment<'tcx>>> {
        let mut obligations = vec![];
        let targets = self.steps.iter().skip(1).map(|&(ty, _)| ty)
            .chain(iter::once(self.cur_ty));
        let steps: Vec<_> = self.steps.iter().map(|&(source, kind)| {
            if let AutoderefKind::Overloaded = kind {
                fcx.try_overloaded_deref(self.span, source, needs)
                    .and_then(|InferOk { value: method, obligations: o }| {
                        obligations.extend(o);
                        if let ty::Ref(region, _, mutbl) = method.sig.output().sty {
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
    /// e.g., assuming ptr_to_Foo is the type `*const Foo`
    /// fcx.autoderef(span, ptr_to_Foo)  => [*const Foo]
    /// fcx.autoderef(span, ptr_to_Foo).include_raw_ptrs() => [*const Foo, Foo]
    pub fn include_raw_pointers(mut self) -> Self {
        self.include_raw_pointers = true;
        self
    }

    pub fn silence_errors(mut self) -> Self {
        self.silence_errors = true;
        self
    }

    pub fn reached_recursion_limit(&self) -> bool {
        self.reached_recursion_limit
    }

    pub fn finalize(self, fcx: &FnCtxt<'a, 'tcx>) {
        fcx.register_predicates(self.into_obligations());
    }

    pub fn into_obligations(self) -> Vec<traits::PredicateObligation<'tcx>> {
        self.obligations
    }
}

pub fn report_autoderef_recursion_limit_error<'tcx>(tcx: TyCtxt<'tcx>, span: Span, ty: Ty<'tcx>) {
    // We've reached the recursion limit, error gracefully.
    let suggested_limit = *tcx.sess.recursion_limit.get() * 2;
    let msg = format!("reached the recursion limit while auto-dereferencing `{:?}`",
                      ty);
    let error_id = (DiagnosticMessageId::ErrorId(55), Some(span), msg);
    let fresh = tcx.sess.one_time_diagnostics.borrow_mut().insert(error_id);
    if fresh {
        struct_span_err!(tcx.sess,
                         span,
                         E0055,
                         "reached the recursion limit while auto-dereferencing `{:?}`",
                         ty)
            .span_label(span, "deref recursion limit reached")
            .help(&format!(
                "consider adding a `#![recursion_limit=\"{}\"]` attribute to your crate",
                suggested_limit))
            .emit();
    }
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub fn autoderef(&'a self, span: Span, base_ty: Ty<'tcx>) -> Autoderef<'a, 'tcx> {
        Autoderef::new(self, self.param_env, self.body_id, span, base_ty)
    }

    pub fn try_overloaded_deref(&self,
                                span: Span,
                                base_ty: Ty<'tcx>,
                                needs: Needs)
                                -> Option<InferOk<'tcx, MethodCallee<'tcx>>> {
        self.try_overloaded_place_op(span, base_ty, &[], needs, PlaceOp::Deref)
    }
}
