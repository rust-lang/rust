//! Autoderef helpers for inference.

use std::iter;

use rustc_ast_ir::Mutability;

use crate::{
    Adjust, Adjustment, OverloadedDeref,
    autoderef::{Autoderef, AutoderefKind},
    infer::unify::InferenceTable,
    next_solver::{
        Ty,
        infer::{InferOk, traits::PredicateObligations},
    },
};

impl<'db> InferenceTable<'db> {
    pub(crate) fn autoderef(&mut self, base_ty: Ty<'db>) -> Autoderef<'_, 'db> {
        Autoderef::new(self, base_ty)
    }
}

impl<'db> Autoderef<'_, 'db> {
    /// Returns the adjustment steps.
    pub(crate) fn adjust_steps(mut self) -> Vec<Adjustment<'db>> {
        let infer_ok = self.adjust_steps_as_infer_ok();
        self.table.register_infer_ok(infer_ok)
    }

    pub(crate) fn adjust_steps_as_infer_ok(&mut self) -> InferOk<'db, Vec<Adjustment<'db>>> {
        let steps = self.steps();
        if steps.is_empty() {
            return InferOk { obligations: PredicateObligations::new(), value: vec![] };
        }

        let targets = steps.iter().skip(1).map(|&(ty, _)| ty).chain(iter::once(self.final_ty()));
        let steps: Vec<_> = steps
            .iter()
            .map(|&(_source, kind)| {
                if let AutoderefKind::Overloaded = kind {
                    Some(OverloadedDeref(Some(Mutability::Not)))
                } else {
                    None
                }
            })
            .zip(targets)
            .map(|(autoderef, target)| Adjustment { kind: Adjust::Deref(autoderef), target })
            .collect();

        InferOk { obligations: self.take_obligations(), value: steps }
    }
}
