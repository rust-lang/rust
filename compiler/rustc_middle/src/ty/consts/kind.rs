use super::Const;
use crate::mir;
use crate::ty::abstract_const::CastKind;
use crate::ty::{self, visit::TypeVisitableExt as _, List, Ty, TyCtxt};
use rustc_macros::{extension, HashStable, TyDecodable, TyEncodable, TypeFoldable, TypeVisitable};

#[extension(pub(crate) trait UnevaluatedConstEvalExt<'tcx>)]
impl<'tcx> ty::UnevaluatedConst<'tcx> {
    /// FIXME(RalfJung): I cannot explain what this does or why it makes sense, but not doing this
    /// hurts performance.
    #[inline]
    fn prepare_for_eval(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> (ty::ParamEnv<'tcx>, Self) {
        // HACK(eddyb) this erases lifetimes even though `const_eval_resolve`
        // also does later, but we want to do it before checking for
        // inference variables.
        // Note that we erase regions *before* calling `with_reveal_all_normalized`,
        // so that we don't try to invoke this query with
        // any region variables.

        // HACK(eddyb) when the query key would contain inference variables,
        // attempt using identity args and `ParamEnv` instead, that will succeed
        // when the expression doesn't depend on any parameters.
        // FIXME(eddyb, skinny121) pass `InferCtxt` into here when it's available, so that
        // we can call `infcx.const_eval_resolve` which handles inference variables.
        if (param_env, self).has_non_region_infer() {
            (
                tcx.param_env(self.def),
                ty::UnevaluatedConst {
                    def: self.def,
                    args: ty::GenericArgs::identity_for_item(tcx, self.def),
                },
            )
        } else {
            (tcx.erase_regions(param_env).with_reveal_all_normalized(tcx), tcx.erase_regions(self))
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
#[derive(HashStable, TyEncodable, TyDecodable, TypeVisitable, TypeFoldable)]
pub enum Expr<'tcx> {
    Binop(mir::BinOp, Const<'tcx>, Const<'tcx>),
    UnOp(mir::UnOp, Const<'tcx>),
    FunctionCall(Const<'tcx>, &'tcx List<Const<'tcx>>),
    Cast(CastKind, Const<'tcx>, Ty<'tcx>),
}

#[cfg(target_pointer_width = "64")]
rustc_data_structures::static_assert_size!(Expr<'_>, 24);
