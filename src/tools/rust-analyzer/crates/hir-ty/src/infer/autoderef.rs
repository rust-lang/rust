//! Autoderef helpers for inference.

use std::iter;

use rustc_ast_ir::Mutability;

use crate::{
    Adjust, Adjustment, OverloadedDeref,
    autoderef::{Autoderef, AutoderefCtx, AutoderefKind, GeneralAutoderef},
    infer::unify::InferenceTable,
    next_solver::{
        Ty,
        infer::{InferOk, traits::PredicateObligations},
    },
};

impl<'db> InferenceTable<'db> {
    pub(crate) fn autoderef(&self, base_ty: Ty<'db>) -> Autoderef<'_, 'db, usize> {
        Autoderef::new(&self.infer_ctxt, self.param_env, base_ty)
    }

    pub(crate) fn autoderef_with_tracking(&self, base_ty: Ty<'db>) -> Autoderef<'_, 'db> {
        Autoderef::new_with_tracking(&self.infer_ctxt, self.param_env, base_ty)
    }
}

impl<'db, Ctx: AutoderefCtx<'db>> GeneralAutoderef<'db, Ctx> {
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
