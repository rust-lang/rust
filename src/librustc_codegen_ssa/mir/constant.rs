use rustc::mir::interpret::ErrorHandled;
use rustc::mir;
use rustc_data_structures::indexed_vec::Idx;
use rustc::ty::{self, Ty};
use rustc::ty::layout::{self, HasTyCtxt};
use syntax::source_map::Span;
use crate::traits::*;

use super::FunctionCx;

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    pub fn eval_mir_constant(
        &mut self,
        constant: &mir::Constant<'tcx>,
    ) -> Result<&'tcx ty::Const<'tcx>, ErrorHandled> {
        match constant.literal.val {
            mir::interpret::ConstValue::Unevaluated(def_id, ref substs) => {
                let substs = self.monomorphize(substs);
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
                let fields = match c.ty.sty {
                    ty::Array(_, n) => n.unwrap_usize(bx.tcx()),
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
