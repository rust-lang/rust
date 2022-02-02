use crate::mir::operand::OperandRef;
use crate::traits::*;
use rustc_middle::mir;
use rustc_middle::mir::interpret::{ConstValue, ErrorHandled};
use rustc_middle::ty::layout::HasTyCtxt;
use rustc_middle::ty::{self, Ty};
use rustc_span::source_map::Span;
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
        let ct = match ct {
            mir::ConstantKind::Ty(ct) => ct,
            mir::ConstantKind::Val(val, _) => return Ok(val),
        };
        match ct.val() {
            ty::ConstKind::Unevaluated(ct) => self
                .cx
                .tcx()
                .const_eval_resolve(ty::ParamEnv::reveal_all(), ct, None)
                .map_err(|err| {
                    self.cx.tcx().sess.span_err(constant.span, "erroneous constant encountered");
                    err
                }),
            ty::ConstKind::Value(value) => Ok(value),
            err => span_bug!(
                constant.span,
                "encountered bad ConstKind after monomorphizing: {:?}",
                err
            ),
        }
    }

    /// process constant containing SIMD shuffle indices
    pub fn simd_shuffle_indices(
        &mut self,
        bx: &Bx,
        span: Span,
        ty: Ty<'tcx>,
        constant: Result<ConstValue<'tcx>, ErrorHandled>,
    ) -> (Bx::Value, Ty<'tcx>) {
        constant
            .map(|val| {
                let field_ty = ty.builtin_index().unwrap();
                let c = ty::Const::from_value(bx.tcx(), val, ty);
                let values: Vec<_> = bx
                    .tcx()
                    .destructure_const(ty::ParamEnv::reveal_all().and(c))
                    .fields
                    .iter()
                    .map(|field| {
                        if let Some(prim) = field.val().try_to_scalar() {
                            let layout = bx.layout_of(field_ty);
                            let scalar = match layout.abi {
                                Abi::Scalar(x) => x,
                                _ => bug!("from_const: invalid ByVal layout: {:#?}", layout),
                            };
                            bx.scalar_to_backend(prim, scalar, bx.immediate_backend_type(layout))
                        } else {
                            bug!("simd shuffle field {:?}", field)
                        }
                    })
                    .collect();
                let llval = bx.const_struct(&values, false);
                (llval, c.ty())
            })
            .unwrap_or_else(|_| {
                bx.tcx().sess.span_err(span, "could not evaluate shuffle_indices at compile time");
                // We've errored, so we don't have to produce working code.
                let ty = self.monomorphize(ty);
                let llty = bx.backend_type(bx.layout_of(ty));
                (bx.const_undef(llty), ty)
            })
    }
}
