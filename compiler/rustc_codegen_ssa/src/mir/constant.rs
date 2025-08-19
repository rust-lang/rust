use rustc_abi::BackendRepr;
use rustc_middle::mir::interpret::ErrorHandled;
use rustc_middle::ty::layout::{HasTyCtxt, HasTypingEnv};
use rustc_middle::ty::{self, Ty};
use rustc_middle::{bug, mir, span_bug};

use super::FunctionCx;
use crate::errors;
use crate::mir::operand::OperandRef;
use crate::traits::*;

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    pub(crate) fn eval_mir_constant_to_operand(
        &self,
        bx: &mut Bx,
        constant: &mir::ConstOperand<'tcx>,
    ) -> OperandRef<'tcx, Bx::Value> {
        let val = self.eval_mir_constant(constant);
        let ty = self.monomorphize(constant.ty());
        OperandRef::from_const(bx, val, ty)
    }

    pub fn eval_mir_constant(&self, constant: &mir::ConstOperand<'tcx>) -> mir::ConstValue {
        // `MirUsedCollector` visited all required_consts before codegen began, so if we got here
        // there can be no more constants that fail to evaluate.
        self.monomorphize(constant.const_)
            .eval(self.cx.tcx(), self.cx.typing_env(), constant.span)
            .expect("erroneous constant missed by mono item collection")
    }

    /// This is a convenience helper for `immediate_const_vector`. It has the precondition
    /// that the given `constant` is an `Const::Unevaluated` and must be convertible to
    /// a `ValTree`. If you want a more general version of this, talk to `wg-const-eval` on zulip.
    ///
    /// Note that this function is cursed, since usually MIR consts should not be evaluated to
    /// valtrees!
    fn eval_unevaluated_mir_constant_to_valtree(
        &self,
        constant: &mir::ConstOperand<'tcx>,
    ) -> Result<Result<ty::ValTree<'tcx>, Ty<'tcx>>, ErrorHandled> {
        let uv = match self.monomorphize(constant.const_) {
            mir::Const::Unevaluated(uv, _) => uv.shrink(),
            mir::Const::Ty(_, c) => match c.kind() {
                // A constant that came from a const generic but was then used as an argument to
                // old-style simd_shuffle (passing as argument instead of as a generic param).
                ty::ConstKind::Value(cv) => return Ok(Ok(cv.valtree)),
                other => span_bug!(constant.span, "{other:#?}"),
            },
            // We should never encounter `Const::Val` unless MIR opts (like const prop) evaluate
            // a constant and write that value back into `Operand`s. This could happen, but is
            // unlikely. Also: all users of `simd_shuffle` are on unstable and already need to take
            // a lot of care around intrinsics. For an issue to happen here, it would require a
            // macro expanding to a `simd_shuffle` call without wrapping the constant argument in a
            // `const {}` block, but the user pass through arbitrary expressions.
            // FIXME(oli-obk): replace the magic const generic argument of `simd_shuffle` with a
            // real const generic, and get rid of this entire function.
            other => span_bug!(constant.span, "{other:#?}"),
        };
        let uv = self.monomorphize(uv);
        self.cx.tcx().const_eval_resolve_for_typeck(self.cx.typing_env(), uv, constant.span)
    }

    /// process constant containing SIMD shuffle indices & constant vectors
    pub fn immediate_const_vector(
        &mut self,
        bx: &Bx,
        constant: &mir::ConstOperand<'tcx>,
    ) -> (Bx::Value, Ty<'tcx>) {
        let ty = self.monomorphize(constant.ty());
        assert!(ty.is_simd());
        let field_ty = ty.simd_size_and_type(bx.tcx()).1;

        let val = self
            .eval_unevaluated_mir_constant_to_valtree(constant)
            .ok()
            .map(|x| x.ok())
            .flatten()
            .map(|val| {
                // A SIMD type has a single field, which is an array.
                let fields = val.unwrap_branch();
                assert_eq!(fields.len(), 1);
                let array = fields[0].unwrap_branch();
                // Iterate over the array elements to obtain the values in the vector.
                let values: Vec<_> = array
                    .iter()
                    .map(|field| {
                        if let Some(prim) = field.try_to_scalar() {
                            let layout = bx.layout_of(field_ty);
                            let BackendRepr::Scalar(scalar) = layout.backend_repr else {
                                bug!("from_const: invalid ByVal layout: {:#?}", layout);
                            };
                            bx.scalar_to_backend(prim, scalar, bx.immediate_backend_type(layout))
                        } else {
                            bug!("field is not a scalar {:?}", field)
                        }
                    })
                    .collect();
                bx.const_vector(&values)
            })
            .unwrap_or_else(|| {
                bx.tcx().dcx().emit_err(errors::ShuffleIndicesEvaluation { span: constant.span });
                // We've errored, so we don't have to produce working code.
                let llty = bx.backend_type(bx.layout_of(ty));
                bx.const_undef(llty)
            });
        (val, ty)
    }
}
