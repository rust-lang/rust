//! Some helper functions for `AutoDeref`.

use std::iter;

use itertools::Itertools;
use rustc_hir_analysis::autoderef::{Autoderef, AutoderefKind};
use rustc_infer::infer::InferOk;
use rustc_infer::traits::PredicateObligations;
use rustc_middle::ty::adjustment::{Adjust, Adjustment, OverloadedDeref};
use rustc_middle::ty::{self, Ty};
use rustc_span::Span;

use super::method::MethodCallee;
use super::{FnCtxt, PlaceOp};

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub(crate) fn autoderef(&'a self, span: Span, base_ty: Ty<'tcx>) -> Autoderef<'a, 'tcx> {
        Autoderef::new(self, self.param_env, self.body_id, span, base_ty)
    }

    pub(crate) fn try_overloaded_deref(
        &self,
        span: Span,
        base_ty: Ty<'tcx>,
    ) -> Option<InferOk<'tcx, MethodCallee<'tcx>>> {
        self.try_overloaded_place_op(span, base_ty, None, PlaceOp::Deref)
    }

    /// Returns the adjustment steps.
    pub(crate) fn adjust_steps(&self, autoderef: &Autoderef<'a, 'tcx>) -> Vec<Adjustment<'tcx>> {
        self.register_infer_ok_obligations(self.adjust_steps_as_infer_ok(autoderef))
    }

    pub(crate) fn adjust_steps_as_infer_ok(
        &self,
        autoderef: &Autoderef<'a, 'tcx>,
    ) -> InferOk<'tcx, Vec<Adjustment<'tcx>>> {
        let steps = autoderef.steps();
        if steps.is_empty() {
            return InferOk { obligations: PredicateObligations::new(), value: vec![] };
        }

        let mut obligations = PredicateObligations::new();
        let targets =
            steps.iter().skip(1).map(|&(ty, _)| ty).chain(iter::once(autoderef.final_ty()));
        let steps: Vec<_> = steps
            .iter()
            .map(|&(source, kind)| {
                if let AutoderefKind::Overloaded = kind {
                    self.try_overloaded_deref(autoderef.span(), source).and_then(
                        |InferOk { value: method, obligations: o }| {
                            obligations.extend(o);
                            // FIXME: we should assert the sig is &T here... there's no reason for this to be fallible.
                            if let ty::Ref(_, _, mutbl) = *method.sig.output().kind() {
                                Some(OverloadedDeref { mutbl, span: autoderef.span() })
                            } else {
                                None
                            }
                        },
                    )
                } else {
                    None
                }
            })
            .zip_eq(targets)
            .map(|(autoderef, target)| Adjustment { kind: Adjust::Deref(autoderef), target })
            .collect();

        InferOk { obligations, value: steps }
    }
}
