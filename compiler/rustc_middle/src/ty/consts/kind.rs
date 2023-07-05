use super::Const;
use crate::mir;
use crate::ty::abstract_const::CastKind;
use crate::ty::subst::SubstsRef;
use crate::ty::{self, List, Ty};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hir::def_id::DefId;
use rustc_macros::HashStable;

/// An unevaluated (potentially generic) constant used in the type-system.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord, TyEncodable, TyDecodable, Lift)]
#[derive(Hash, HashStable, TypeFoldable, TypeVisitable)]
pub struct UnevaluatedConst<'tcx> {
    pub def: DefId,
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
    pub fn new(def: DefId, substs: SubstsRef<'tcx>) -> UnevaluatedConst<'tcx> {
        UnevaluatedConst { def, substs }
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
static_assert_size!(super::ConstKind<'_>, 32);

/// An inference variable for a const, for use in const generics.
#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, TyEncodable, TyDecodable, Hash)]
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
