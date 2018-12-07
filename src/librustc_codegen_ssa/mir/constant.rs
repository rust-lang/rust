use rustc::mir::interpret::ErrorHandled;
use rustc_mir::const_eval::const_field;
use rustc::mir;
use rustc_data_structures::indexed_vec::Idx;
use rustc::mir::interpret::GlobalId;
use rustc::ty::{self, Ty};
use rustc::ty::layout;
use syntax::source_map::Span;
use crate::traits::*;

use super::FunctionCx;

impl<'a, 'tcx: 'a, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    fn fully_evaluate(
        &mut self,
        bx: &Bx,
        constant: &'tcx ty::LazyConst<'tcx>,
    ) -> Result<ty::Const<'tcx>, ErrorHandled> {
        match *constant {
            ty::LazyConst::Unevaluated(def_id, ref substs) => {
                let tcx = bx.tcx();
                let param_env = ty::ParamEnv::reveal_all();
                let instance = ty::Instance::resolve(tcx, param_env, def_id, substs).unwrap();
                let cid = GlobalId {
                    instance,
                    promoted: None,
                };
                tcx.const_eval(param_env.and(cid))
            },
            ty::LazyConst::Evaluated(constant) => Ok(constant),
        }
    }

    pub fn eval_mir_constant(
        &mut self,
        bx: &Bx,
        constant: &mir::Constant<'tcx>,
    ) -> Result<ty::Const<'tcx>, ErrorHandled> {
        let c = self.monomorphize(&constant.literal);
        self.fully_evaluate(bx, c)
    }

    /// process constant containing SIMD shuffle indices
    pub fn simd_shuffle_indices(
        &mut self,
        bx: &Bx,
        span: Span,
        ty: Ty<'tcx>,
        constant: Result<ty::Const<'tcx>, ErrorHandled>,
    ) -> (Bx::Value, Ty<'tcx>) {
        constant
            .map(|c| {
                let field_ty = c.ty.builtin_index().unwrap();
                let fields = match c.ty.sty {
                    ty::Array(_, n) => n.unwrap_usize(bx.tcx()),
                    _ => bug!("invalid simd shuffle type: {}", c.ty),
                };
                let values: Vec<_> = (0..fields).map(|field| {
                    let field = const_field(
                        bx.tcx(),
                        ty::ParamEnv::reveal_all(),
                        None,
                        mir::Field::new(field as usize),
                        c,
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
