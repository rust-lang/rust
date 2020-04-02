use crate::mir::operand::OperandRef;
use crate::traits::*;
use rustc_index::vec::Idx;
use rustc_middle::mir;
use rustc_middle::mir::interpret::{ConstValue, ErrorHandled};
use rustc_middle::ty::layout::HasTyCtxt;
use rustc_middle::ty::{self, Ty};
use rustc_span::source_map::Span;
use rustc_target::abi::Abi;

use super::FunctionCx;

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    pub fn eval_mir_constant_to_operand(
        &mut self,
        bx: &mut Bx,
        constant: &mir::Constant<'tcx>,
    ) -> Result<OperandRef<'tcx, Bx::Value>, ErrorHandled> {
        match constant.literal.val {
            // Special case unevaluated statics, because statics have an identity and thus should
            // use `get_static` to get at their id.
            // FIXME(oli-obk): can we unify this somehow, maybe by making const eval of statics
            // always produce `&STATIC`. This may also simplify how const eval works with statics.
            ty::ConstKind::Unevaluated(def_id, substs, None) if self.cx.tcx().is_static(def_id) => {
                assert!(substs.is_empty(), "we don't support generic statics yet");
                let static_ = bx.get_static(def_id);
                // we treat operands referring to statics as if they were `&STATIC` instead
                let ptr_ty = self.cx.tcx().mk_mut_ptr(self.monomorphize(&constant.literal.ty));
                let layout = bx.layout_of(ptr_ty);
                Ok(OperandRef::from_immediate_or_packed_pair(bx, static_, layout))
            }
            _ => {
                let val = self.eval_mir_constant(constant)?;
                let ty = self.monomorphize(&constant.literal.ty);
                Ok(OperandRef::from_const(bx, val, ty))
            }
        }
    }

    pub fn eval_mir_constant(
        &mut self,
        constant: &mir::Constant<'tcx>,
    ) -> Result<ConstValue<'tcx>, ErrorHandled> {
        match self.monomorphize(&constant.literal).val {
            ty::ConstKind::Unevaluated(def_id, substs, promoted) => self
                .cx
                .tcx()
                .const_eval_resolve(ty::ParamEnv::reveal_all(), def_id, substs, promoted, None)
                .map_err(|err| {
                    if promoted.is_none() {
                        self.cx
                            .tcx()
                            .sess
                            .span_err(constant.span, "erroneous constant encountered");
                    }
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
                let fields = match ty.kind {
                    ty::Array(_, n) => n.eval_usize(bx.tcx(), ty::ParamEnv::reveal_all()),
                    _ => bug!("invalid simd shuffle type: {}", ty),
                };
                let c = ty::Const::from_value(bx.tcx(), val, ty);
                let values: Vec<_> = (0..fields)
                    .map(|field| {
                        let field = bx.tcx().const_field(
                            ty::ParamEnv::reveal_all().and((&c, mir::Field::new(field as usize))),
                        );
                        if let Some(prim) = field.try_to_scalar() {
                            let layout = bx.layout_of(field_ty);
                            let scalar = match layout.abi {
                                Abi::Scalar(ref x) => x,
                                _ => bug!("from_const: invalid ByVal layout: {:#?}", layout),
                            };
                            bx.scalar_to_backend(prim, scalar, bx.immediate_backend_type(layout))
                        } else {
                            bug!("simd shuffle field {:?}", field)
                        }
                    })
                    .collect();
                let llval = bx.const_struct(&values, false);
                (llval, c.ty)
            })
            .unwrap_or_else(|_| {
                bx.tcx().sess.span_err(span, "could not evaluate shuffle_indices at compile time");
                // We've errored, so we don't have to produce working code.
                let ty = self.monomorphize(&ty);
                let llty = bx.backend_type(bx.layout_of(ty));
                (bx.const_undef(llty), ty)
            })
    }
}
