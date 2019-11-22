use rustc::mir::interpret::ErrorHandled;
use rustc::mir;
use rustc_index::vec::Idx;
use rustc::ty::{self, Ty};
use rustc::ty::layout::{self, HasTyCtxt};
use syntax::source_map::Span;
use crate::traits::*;
use crate::mir::operand::OperandRef;

use super::FunctionCx;

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    pub fn eval_mir_constant_to_operand(
        &mut self,
        bx: &mut Bx,
        constant: &mir::Constant<'tcx>,
    ) -> Result<OperandRef<'tcx, Bx::Value>, ErrorHandled> {
        match constant.literal.val {
            ty::ConstKind::Unevaluated(def_id, substs)
                if self.cx.tcx().is_static(def_id) => {
                    assert!(substs.is_empty(), "we don't support generic statics yet");
                    let static_ = bx.get_static(def_id);
                    // we treat operands referring to statics as if they were `&STATIC` instead
                    let ptr_ty = self.cx.tcx().mk_mut_ptr(self.monomorphize(&constant.literal.ty));
                    let layout = bx.layout_of(ptr_ty);
                    Ok(OperandRef::from_immediate_or_packed_pair(bx, static_, layout))
                }
            _ => {
                let val = self.eval_mir_constant(constant)?;
                Ok(OperandRef::from_const(bx, val))
            }
        }
    }

    pub fn eval_mir_constant(
        &mut self,
        constant: &mir::Constant<'tcx>,
    ) -> Result<&'tcx ty::Const<'tcx>, ErrorHandled> {
        match constant.literal.val {
            ty::ConstKind::Unevaluated(def_id, substs) => {
                let substs = self.monomorphize(&substs);
                let instance = ty::Instance::resolve(
                    self.cx.tcx(), ty::ParamEnv::reveal_all(), def_id, substs,
                ).unwrap();
                let cid = mir::interpret::GlobalId {
                    instance,
                    promoted: None,
                };
                self.cx.tcx().const_eval(ty::ParamEnv::reveal_all().and(cid))
            },
            _ => Ok(self.monomorphize(&constant.literal)),
        }
    }

    /// process constant containing SIMD shuffle indices
    pub fn simd_shuffle_indices(
        &mut self,
        bx: &Bx,
        span: Span,
        ty: Ty<'tcx>,
        constant: Result<&'tcx ty::Const<'tcx>, ErrorHandled>,
    ) -> (Bx::Value, Ty<'tcx>) {
        constant
            .map(|c| {
                let field_ty = c.ty.builtin_index().unwrap();
                let fields = match c.ty.kind {
                    ty::Array(_, n) => n.eval_usize(bx.tcx(), ty::ParamEnv::reveal_all()),
                    _ => bug!("invalid simd shuffle type: {}", c.ty),
                };
                let values: Vec<_> = (0..fields).map(|field| {
                    let field = bx.tcx().const_field(
                        ty::ParamEnv::reveal_all().and((&c, mir::Field::new(field as usize)))
                    );
                    if let Some(prim) = field.val.try_to_scalar() {
                        let layout = bx.layout_of(field_ty);
                        let scalar = match layout.abi {
                            layout::Abi::Scalar(ref x) => x,
                            _ => bug!("from_const: invalid ByVal layout: {:#?}", layout)
                        };
                        bx.scalar_to_backend(
                            prim, scalar,
                            bx.immediate_backend_type(layout),
                        )
                    } else {
                        bug!("simd shuffle field {:?}", field)
                    }
                }).collect();
                let llval = bx.const_struct(&values, false);
                (llval, c.ty)
            })
            .unwrap_or_else(|_| {
                bx.tcx().sess.span_err(
                    span,
                    "could not evaluate shuffle_indices at compile time",
                );
                // We've errored, so we don't have to produce working code.
                let ty = self.monomorphize(&ty);
                let llty = bx.backend_type(bx.layout_of(ty));
                (bx.const_undef(llty), ty)
            })
    }
}
