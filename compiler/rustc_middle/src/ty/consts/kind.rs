use super::Const;
use crate::mir;
use crate::mir::interpret::{AllocId, ConstValue, Scalar};
use crate::ty::abstract_const::CastKind;
use crate::ty::subst::{InternalSubsts, SubstsRef};
use crate::ty::ParamEnv;
use crate::ty::{self, List, Ty, TyCtxt, TypeVisitable};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_errors::ErrorGuaranteed;
use rustc_hir::def_id::DefId;
use rustc_macros::HashStable;
use rustc_target::abi::Size;

use super::ScalarInt;

/// An unevaluated (potentially generic) constant used in the type-system.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord, TyEncodable, TyDecodable, Lift)]
#[derive(Hash, HashStable, TypeFoldable, TypeVisitable)]
pub struct UnevaluatedConst<'tcx> {
    pub def: ty::WithOptConstParam<DefId>,
    pub substs: SubstsRef<'tcx>,
}

impl rustc_errors::IntoDiagnosticArg for UnevaluatedConst<'_> {
    fn into_diagnostic_arg(self) -> rustc_errors::DiagnosticArgValue<'static> {
        format!("{:?}", self).into_diagnostic_arg()
    }
}

impl<'tcx> UnevaluatedConst<'tcx> {
    #[inline]
    pub fn expand(self) -> mir::UnevaluatedConst<'tcx> {
        mir::UnevaluatedConst { def: self.def, substs: self.substs, promoted: None }
    }
}

impl<'tcx> UnevaluatedConst<'tcx> {
    #[inline]
    pub fn new(
        def: ty::WithOptConstParam<DefId>,
        substs: SubstsRef<'tcx>,
    ) -> UnevaluatedConst<'tcx> {
        UnevaluatedConst { def, substs }
    }
}

/// Represents a constant in Rust.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord, TyEncodable, TyDecodable)]
#[derive(Hash, HashStable, TypeFoldable, TypeVisitable)]
#[derive(derive_more::From)]
pub enum ConstKind<'tcx> {
    /// A const generic parameter.
    Param(ty::ParamConst),

    /// Infer the value of the const.
    Infer(InferConst<'tcx>),

    /// Bound const variable, used only when preparing a trait query.
    Bound(ty::DebruijnIndex, ty::BoundVar),

    /// A placeholder const - universally quantified higher-ranked const.
    Placeholder(ty::PlaceholderConst<'tcx>),

    /// Used in the HIR by using `Unevaluated` everywhere and later normalizing to one of the other
    /// variants when the code is monomorphic enough for that.
    Unevaluated(UnevaluatedConst<'tcx>),

    /// Used to hold computed value.
    Value(ty::ValTree<'tcx>),

    /// A placeholder for a const which could not be computed; this is
    /// propagated to avoid useless error messages.
    #[from(ignore)]
    Error(ErrorGuaranteed),

    /// Expr which contains an expression which has partially evaluated items.
    Expr(Expr<'tcx>),
}

impl<'tcx> From<ty::ConstVid<'tcx>> for ConstKind<'tcx> {
    fn from(const_vid: ty::ConstVid<'tcx>) -> Self {
        InferConst::Var(const_vid).into()
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord, Hash)]
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
static_assert_size!(ConstKind<'_>, 32);

impl<'tcx> ConstKind<'tcx> {
    #[inline]
    pub fn try_to_value(self) -> Option<ty::ValTree<'tcx>> {
        if let ConstKind::Value(val) = self { Some(val) } else { None }
    }

    #[inline]
    pub fn try_to_scalar(self) -> Option<Scalar<AllocId>> {
        self.try_to_value()?.try_to_scalar()
    }

    #[inline]
    pub fn try_to_scalar_int(self) -> Option<ScalarInt> {
        self.try_to_value()?.try_to_scalar_int()
    }

    #[inline]
    pub fn try_to_bits(self, size: Size) -> Option<u128> {
        self.try_to_scalar_int()?.to_bits(size).ok()
    }

    #[inline]
    pub fn try_to_bool(self) -> Option<bool> {
        self.try_to_scalar_int()?.try_into().ok()
    }

    #[inline]
    pub fn try_to_target_usize(self, tcx: TyCtxt<'tcx>) -> Option<u64> {
        self.try_to_value()?.try_to_target_usize(tcx)
    }
}

/// An inference variable for a const, for use in const generics.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord, TyEncodable, TyDecodable, Hash)]
pub enum InferConst<'tcx> {
    /// Infer the value of the const.
    Var(ty::ConstVid<'tcx>),
    /// A fresh const variable. See `infer::freshen` for more details.
    Fresh(u32),
}

impl<CTX> HashStable<CTX> for InferConst<'_> {
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        match self {
            InferConst::Var(_) => panic!("const variables should not be hashed: {self:?}"),
            InferConst::Fresh(i) => i.hash_stable(hcx, hasher),
        }
    }
}

enum EvalMode {
    Typeck,
    Mir,
}

enum EvalResult<'tcx> {
    ValTree(ty::ValTree<'tcx>),
    ConstVal(ConstValue<'tcx>),
}

impl<'tcx> ConstKind<'tcx> {
    #[inline]
    /// Tries to evaluate the constant if it is `Unevaluated`. If that doesn't succeed, return the
    /// unevaluated constant.
    pub fn eval(self, tcx: TyCtxt<'tcx>, param_env: ParamEnv<'tcx>) -> Self {
        self.try_eval_for_typeck(tcx, param_env).and_then(Result::ok).map_or(self, ConstKind::Value)
    }

    #[inline]
    /// Tries to evaluate the constant if it is `Unevaluated`. If that isn't possible or necessary
    /// return `None`.
    // FIXME(@lcnr): Completely rework the evaluation/normalization system for `ty::Const` once valtrees are merged.
    pub fn try_eval_for_mir(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
    ) -> Option<Result<ConstValue<'tcx>, ErrorGuaranteed>> {
        match self.try_eval_inner(tcx, param_env, EvalMode::Mir) {
            Some(Ok(EvalResult::ValTree(_))) => unreachable!(),
            Some(Ok(EvalResult::ConstVal(v))) => Some(Ok(v)),
            Some(Err(e)) => Some(Err(e)),
            None => None,
        }
    }

    #[inline]
    /// Tries to evaluate the constant if it is `Unevaluated`. If that isn't possible or necessary
    /// return `None`.
    // FIXME(@lcnr): Completely rework the evaluation/normalization system for `ty::Const` once valtrees are merged.
    pub fn try_eval_for_typeck(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
    ) -> Option<Result<ty::ValTree<'tcx>, ErrorGuaranteed>> {
        match self.try_eval_inner(tcx, param_env, EvalMode::Typeck) {
            Some(Ok(EvalResult::ValTree(v))) => Some(Ok(v)),
            Some(Ok(EvalResult::ConstVal(_))) => unreachable!(),
            Some(Err(e)) => Some(Err(e)),
            None => None,
        }
    }

    #[inline]
    fn try_eval_inner(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
        eval_mode: EvalMode,
    ) -> Option<Result<EvalResult<'tcx>, ErrorGuaranteed>> {
        assert!(!self.has_escaping_bound_vars(), "escaping vars in {self:?}");
        if let ConstKind::Unevaluated(unevaluated) = self {
            use crate::mir::interpret::ErrorHandled;

            // HACK(eddyb) this erases lifetimes even though `const_eval_resolve`
            // also does later, but we want to do it before checking for
            // inference variables.
            // Note that we erase regions *before* calling `with_reveal_all_normalized`,
            // so that we don't try to invoke this query with
            // any region variables.
            let param_env_and = tcx
                .erase_regions(param_env)
                .with_reveal_all_normalized(tcx)
                .and(tcx.erase_regions(unevaluated));

            // HACK(eddyb) when the query key would contain inference variables,
            // attempt using identity substs and `ParamEnv` instead, that will succeed
            // when the expression doesn't depend on any parameters.
            // FIXME(eddyb, skinny121) pass `InferCtxt` into here when it's available, so that
            // we can call `infcx.const_eval_resolve` which handles inference variables.
            let param_env_and = if param_env_and.needs_infer() {
                tcx.param_env(unevaluated.def.did).and(ty::UnevaluatedConst {
                    def: unevaluated.def,
                    substs: InternalSubsts::identity_for_item(tcx, unevaluated.def.did),
                })
            } else {
                param_env_and
            };

            // FIXME(eddyb) maybe the `const_eval_*` methods should take
            // `ty::ParamEnvAnd` instead of having them separate.
            let (param_env, unevaluated) = param_env_and.into_parts();
            // try to resolve e.g. associated constants to their definition on an impl, and then
            // evaluate the const.
            match eval_mode {
                EvalMode::Typeck => {
                    match tcx.const_eval_resolve_for_typeck(param_env, unevaluated, None) {
                        // NOTE(eddyb) `val` contains no lifetimes/types/consts,
                        // and we use the original type, so nothing from `substs`
                        // (which may be identity substs, see above),
                        // can leak through `val` into the const we return.
                        Ok(val) => Some(Ok(EvalResult::ValTree(val?))),
                        Err(ErrorHandled::TooGeneric) => None,
                        Err(ErrorHandled::Reported(e)) => Some(Err(e)),
                    }
                }
                EvalMode::Mir => {
                    match tcx.const_eval_resolve(param_env, unevaluated.expand(), None) {
                        // NOTE(eddyb) `val` contains no lifetimes/types/consts,
                        // and we use the original type, so nothing from `substs`
                        // (which may be identity substs, see above),
                        // can leak through `val` into the const we return.
                        Ok(val) => Some(Ok(EvalResult::ConstVal(val))),
                        Err(ErrorHandled::TooGeneric) => None,
                        Err(ErrorHandled::Reported(e)) => Some(Err(e)),
                    }
                }
            }
        } else {
            None
        }
    }
}
