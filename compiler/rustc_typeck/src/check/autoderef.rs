//! Some helper functions for `AutoDeref`
use super::method::MethodCallee;
use super::{FnCtxt, PlaceOp};

use rustc_infer::infer::InferOk;
use rustc_middle::ty::adjustment::{Adjust, Adjustment, OverloadedDeref};
use rustc_middle::ty::{self, Ty};
use rustc_span::Span;
use rustc_trait_selection::autoderef::{Autoderef, AutoderefKind};

use std::iter;

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub fn autoderef(&'a self, span: Span, base_ty: Ty<'tcx>) -> Autoderef<'a, 'tcx> {
        Autoderef::new(self, self.param_env, self.body_id, span, base_ty)
    }

    pub fn try_overloaded_deref(
        &self,
        span: Span,
        base_ty: Ty<'tcx>,
    ) -> Option<InferOk<'tcx, MethodCallee<'tcx>>> {
        self.try_overloaded_place_op(span, base_ty, &[], PlaceOp::Deref)
    }

    /// Returns the adjustment steps.
    pub fn adjust_steps(&self, autoderef: &Autoderef<'a, 'tcx>) -> Vec<Adjustment<'tcx>> {
        self.register_infer_ok_obligations(self.adjust_steps_as_infer_ok(autoderef))
    }

    pub fn adjust_steps_as_infer_ok(
        &self,
        autoderef: &Autoderef<'a, 'tcx>,
    ) -> InferOk<'tcx, Vec<Adjustment<'tcx>>> {
        let mut obligations = vec![];
        let steps = autoderef.steps();
        let targets =
            steps.iter().skip(1).map(|&(ty, _)| ty).chain(iter::once(autoderef.final_ty(false)));
        let steps: Vec<_> = steps
            .iter()
            .map(|&(source, kind)| {
                if let AutoderefKind::Overloaded = kind {
                    self.try_overloaded_deref(autoderef.span(), source).and_then(
                        |InferOk { value: method, obligations: o }| {
                            obligations.extend(o);
                            if let ty::Ref(region, _, mutbl) = *method.sig.output().kind() {
                                Some(OverloadedDeref { region, mutbl })
                            } else {
                                None
                            }
                        },
                    )
                } else {
                    None
                }
            })
            .zip(targets)
            .map(|(autoderef, target)| Adjustment { kind: Adjust::Deref(autoderef), target })
            .collect();

        InferOk { obligations, value: steps }
    }
}
