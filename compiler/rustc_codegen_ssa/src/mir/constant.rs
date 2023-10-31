use crate::errors;
use crate::mir::operand::OperandRef;
use crate::traits::*;
use rustc_middle::mir;
use rustc_middle::mir::interpret::{ConstValue, ErrorHandled};
use rustc_middle::ty::layout::HasTyCtxt;
use rustc_middle::ty::{self, Ty};
use rustc_target::abi::Abi;

use super::FunctionCx;

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    pub fn eval_mir_constant_to_operand(
        &self,
        bx: &mut Bx,
        constant: &mir::Constant<'tcx>,
    ) -> Result<OperandRef<'tcx, Bx::Value>, ErrorHandled> {
        let val = self.eval_mir_constant(constant)?;
        let ty = self.monomorphize(constant.ty());
        Ok(OperandRef::from_const(bx, val, ty))
    }

    pub fn eval_mir_constant(
        &self,
        constant: &mir::Constant<'tcx>,
    ) -> Result<ConstValue<'tcx>, ErrorHandled> {
        let ct = self.monomorphize(constant.literal);
        let uv = match ct {
            mir::ConstantKind::Ty(ct) => match ct.kind() {
                ty::ConstKind::Unevaluated(uv) => uv.expand(),
                ty::ConstKind::Value(val) => {
                    return Ok(self.cx.tcx().valtree_to_const_val((ct.ty(), val)));
                }
                err => span_bug!(
                    constant.span,
                    "encountered bad ConstKind after monomorphizing: {:?}",
                    err
                ),
            },
            mir::ConstantKind::Unevaluated(uv, _) => uv,
            mir::ConstantKind::Val(val, _) => return Ok(val),
        };

        self.cx.tcx().const_eval_resolve(ty::ParamEnv::reveal_all(), uv, None).map_err(|err| {
            match err {
                ErrorHandled::Reported(_) => {
                    self.cx.tcx().sess.emit_err(errors::ErroneousConstant { span: constant.span });
                }
                ErrorHandled::TooGeneric => {
                    self.cx
                        .tcx()
                        .sess
                        .diagnostic()
                        .emit_bug(errors::PolymorphicConstantTooGeneric { span: constant.span });
                }
            }
            err
        })
    }

    /// This is a convenience helper for `simd_shuffle_indices`. It has the precondition
    /// that the given `constant` is an `ConstantKind::Unevaluated` and must be convertible to
    /// a `ValTree`. If you want a more general version of this, talk to `wg-const-eval` on zulip.
    pub fn eval_unevaluated_mir_constant_to_valtree(
        &self,
        constant: &mir::Constant<'tcx>,
    ) -> Result<Option<ty::ValTree<'tcx>>, ErrorHandled> {
        let uv = match self.monomorphize(constant.literal) {
            mir::ConstantKind::Unevaluated(uv, _) => uv.shrink(),
            mir::ConstantKind::Ty(c) => match c.kind() {
                // A constant that came from a const generic but was then used as an argument to old-style
                // simd_shuffle (passing as argument instead of as a generic param).
                rustc_type_ir::ConstKind::Value(valtree) => return Ok(Some(valtree)),
                other => span_bug!(constant.span, "{other:#?}"),
            },
            // We should never encounter `ConstantKind::Val` unless MIR opts (like const prop) evaluate
            // a constant and write that value back into `Operand`s. This could happen, but is unlikely.
            // Also: all users of `simd_shuffle` are on unstable and already need to take a lot of care
            // around intrinsics. For an issue to happen here, it would require a macro expanding to a
            // `simd_shuffle` call without wrapping the constant argument in a `const {}` block, but
            // the user pass through arbitrary expressions.
            // FIXME(oli-obk): replace the magic const generic argument of `simd_shuffle` with a real
            // const generic.
            other => span_bug!(constant.span, "{other:#?}"),
        };
        let uv = self.monomorphize(uv);
        self.cx.tcx().const_eval_resolve_for_typeck(
            ty::ParamEnv::reveal_all(),
            uv,
            Some(constant.span),
        )
    }

    /// process constant containing SIMD shuffle indices
    pub fn simd_shuffle_indices(
        &mut self,
        bx: &Bx,
        constant: &mir::Constant<'tcx>,
    ) -> (Bx::Value, Ty<'tcx>) {
        let ty = self.monomorphize(constant.ty());
        let val = self
            .eval_unevaluated_mir_constant_to_valtree(constant)
            .ok()
            .flatten()
            .map(|val| {
                let field_ty = ty.builtin_index().unwrap();
                let values: Vec<_> = val
                    .unwrap_branch()
                    .iter()
                    .map(|field| {
                        if let Some(prim) = field.try_to_scalar() {
                            let layout = bx.layout_of(field_ty);
                            let Abi::Scalar(scalar) = layout.abi else {
                                bug!("from_const: invalid ByVal layout: {:#?}", layout);
                            };
                            bx.scalar_to_backend(prim, scalar, bx.immediate_backend_type(layout))
                        } else {
                            bug!("simd shuffle field {:?}", field)
                        }
                    })
                    .collect();
                bx.const_struct(&values, false)
            })
            .unwrap_or_else(|| {
                bx.tcx().sess.emit_err(errors::ShuffleIndicesEvaluation { span: constant.span });
                // We've errored, so we don't have to produce working code.
                let llty = bx.backend_type(bx.layout_of(ty));
                bx.const_undef(llty)
            });
        (val, ty)
    }
}
