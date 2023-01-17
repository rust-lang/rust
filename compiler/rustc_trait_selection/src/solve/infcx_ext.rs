use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::infer::InferCtxt;
use rustc_middle::ty::Ty;
use rustc_span::DUMMY_SP;

/// Methods used inside of the canonical queries of the solver.
pub(super) trait InferCtxtExt<'tcx> {
    fn next_ty_infer(&self) -> Ty<'tcx>;
}

impl<'tcx> InferCtxtExt<'tcx> for InferCtxt<'tcx> {
    fn next_ty_infer(&self) -> Ty<'tcx> {
        self.next_ty_var(TypeVariableOrigin {
            kind: TypeVariableOriginKind::MiscVariable,
            span: DUMMY_SP,
        })
    }
}
