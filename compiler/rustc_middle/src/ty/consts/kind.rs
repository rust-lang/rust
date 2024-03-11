use super::Const;
use crate::mir;
use crate::ty::abstract_const::CastKind;
use crate::ty::GenericArgsRef;
use crate::ty::{self, visit::TypeVisitableExt as _, List, Ty, TyCtxt};
use rustc_hir::def_id::DefId;
use rustc_macros::HashStable;

/// An unevaluated (potentially generic) constant used in the type-system.
#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, TyEncodable, TyDecodable)]
#[derive(Hash, HashStable, TypeFoldable, TypeVisitable)]
pub struct UnevaluatedConst<'tcx> {
    pub def: DefId,
    pub args: GenericArgsRef<'tcx>,
}

impl rustc_errors::IntoDiagArg for UnevaluatedConst<'_> {
    fn into_diag_arg(self) -> rustc_errors::DiagArgValue {
        format!("{self:?}").into_diag_arg()
    }
}

impl<'tcx> UnevaluatedConst<'tcx> {
    /// FIXME(RalfJung): I cannot explain what this does or why it makes sense, but not doing this
    /// hurts performance.
    #[inline]
    pub(crate) fn prepare_for_eval(
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

impl<'tcx> UnevaluatedConst<'tcx> {
    #[inline]
    pub fn new(def: DefId, args: GenericArgsRef<'tcx>) -> UnevaluatedConst<'tcx> {
        UnevaluatedConst { def, args }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
#[derive(HashStable, TyEncodable, TyDecodable, TypeVisitable, TypeFoldable)]
pub enum Expr<'tcx> {
    Binop(mir::BinOp, Const<'tcx>, Const<'tcx>),
    UnOp(mir::UnOp, Const<'tcx>),
    FunctionCall(Const<'tcx>, &'tcx List<Const<'tcx>>),
    Cast(CastKind, Const<'tcx>, Ty<'tcx>),
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
static_assert_size!(Expr<'_>, 24);

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
static_assert_size!(super::ConstKind<'_>, 32);
