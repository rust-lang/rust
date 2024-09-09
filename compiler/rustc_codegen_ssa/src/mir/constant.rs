use rustc_middle::mir::interpret::ErrorHandled;
use rustc_middle::ty::layout::HasTyCtxt;
use rustc_middle::ty::{self, Ty, ValTree};
use rustc_middle::{bug, mir, span_bug};
use rustc_target::abi::Abi;

use super::FunctionCx;
use crate::errors;
use crate::mir::operand::OperandRef;
use crate::traits::*;

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    pub fn eval_mir_constant_to_operand(
        &self,
        bx: &mut Bx,
        constant: &mir::ConstOperand<'tcx>,
    ) -> OperandRef<'tcx, Bx::Value> {
        let val = self.eval_mir_constant(constant);
        let ty = self.monomorphize(constant.ty());
        OperandRef::from_const(bx, val, ty)
    }

    pub fn eval_mir_constant(&self, constant: &mir::ConstOperand<'tcx>) -> mir::ConstValue<'tcx> {
        // `MirUsedCollector` visited all required_consts before codegen began, so if we got here
        // there can be no more constants that fail to evaluate.
        self.monomorphize(constant.const_)
            .eval(self.cx.tcx(), ty::ParamEnv::reveal_all(), constant.span)
            .expect("erroneous constant missed by mono item collection")
    }

    /// This is a convenience helper for `immediate_const_vector`. It has the precondition
    /// that the given `constant` is an `Const::Unevaluated` and must be convertible to
    /// a `ValTree`. If you want a more general version of this, talk to `wg-const-eval` on zulip.
    ///
    /// Note that this function is cursed, since usually MIR consts should not be evaluated to valtrees!
    pub fn eval_unevaluated_mir_constant_to_valtree(
        &self,
        constant: &mir::ConstOperand<'tcx>,
    ) -> Result<Result<ty::ValTree<'tcx>, Ty<'tcx>>, ErrorHandled> {
        let uv = match self.monomorphize(constant.const_) {
            mir::Const::Unevaluated(uv, _) => uv.shrink(),
            mir::Const::Ty(_, c) => match c.kind() {
                // A constant that came from a const generic but was then used as an argument to old-style
                // simd_shuffle (passing as argument instead of as a generic param).
                rustc_type_ir::ConstKind::Value(_, valtree) => return Ok(Ok(valtree)),
                other => span_bug!(constant.span, "{other:#?}"),
            },
            // We should never encounter `Const::Val` unless MIR opts (like const prop) evaluate
            // a constant and write that value back into `Operand`s. This could happen, but is unlikely.
            // Also: all users of `simd_shuffle` are on unstable and already need to take a lot of care
            // around intrinsics. For an issue to happen here, it would require a macro expanding to a
            // `simd_shuffle` call without wrapping the constant argument in a `const {}` block, but
            // the user pass through arbitrary expressions.
            // FIXME(oli-obk): replace the magic const generic argument of `simd_shuffle` with a real
            // const generic, and get rid of this entire function.
            other => span_bug!(constant.span, "{other:#?}"),
        };
        let uv = self.monomorphize(uv);
        self.cx.tcx().const_eval_resolve_for_typeck(ty::ParamEnv::reveal_all(), uv, constant.span)
    }

    /// process constant containing SIMD shuffle indices & constant vectors
    pub fn immediate_const_vector(
        &mut self,
        bx: &Bx,
        constant: &mir::ConstOperand<'tcx>,
    ) -> (Bx::Value, Ty<'tcx>) {
        let ty = self.monomorphize(constant.ty());
        let ty_is_simd = ty.is_simd();
        // FIXME: ideally we'd assert that this is a SIMD type, but simd_shuffle
        // in its current form relies on a regular array being passed as an
        // immediate argument. This hack can be removed once that is fixed.
        let field_ty = if ty_is_simd {
            ty.simd_size_and_type(bx.tcx()).1
        } else {
            ty.builtin_index().unwrap()
        };

        let val = self
            .eval_unevaluated_mir_constant_to_valtree(constant)
            .ok()
            .map(|x| x.ok())
            .flatten()
            .map(|val| {
                // Depending on whether this is a SIMD type with an array field
                // or a type with many fields (one for each elements), the valtree
                // is either a single branch with N children, or a root node
                // with exactly one child which then in turn has many children.
                // So we look at the first child to determine whether it is a
                // leaf or whether we have to go one more layer down.
                let branch_or_leaf = val.unwrap_branch();
                let first = branch_or_leaf.get(0).unwrap();
                let field_iter = match first {
                    ValTree::Branch(_) => first.unwrap_branch().iter(),
                    ValTree::Leaf(_) => branch_or_leaf.iter(),
                };
                let values: Vec<_> = field_iter
                    .map(|field| {
                        if let Some(prim) = field.try_to_scalar() {
                            let layout = bx.layout_of(field_ty);
                            let Abi::Scalar(scalar) = layout.abi else {
                                bug!("from_const: invalid ByVal layout: {:#?}", layout);
                            };
                            bx.scalar_to_backend(prim, scalar, bx.immediate_backend_type(layout))
                        } else {
                            bug!("field is not a scalar {:?}", field)
                        }
                    })
                    .collect();
                if ty_is_simd { bx.const_vector(&values) } else { bx.const_struct(&values, false) }
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
