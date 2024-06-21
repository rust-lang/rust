#[cfg(feature = "nightly")]
use rustc_macros::{HashStable_NoContext, TyDecodable, TyEncodable};
use rustc_type_ir_macros::{TypeFoldable_Generic, TypeVisitable_Generic};

use crate::inherent::*;
use crate::{self as ty, Interner};

#[derive(derivative::Derivative)]
#[derivative(
    Clone(bound = ""),
    Hash(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = ""),
    Debug(bound = ""),
    Copy(bound = "")
)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(TyEncodable, TyDecodable, HashStable_NoContext))]
pub struct OpaqueTypeKey<I: Interner> {
    pub def_id: I::LocalDefId,
    pub args: I::GenericArgs,
}

impl<I: Interner> OpaqueTypeKey<I> {
    pub fn iter_captured_args(self, tcx: I) -> impl Iterator<Item = (usize, I::GenericArg)> {
        let variances = tcx.variances_of(self.def_id.into());
        std::iter::zip(self.args, variances.into_iter()).enumerate().filter_map(|(i, (arg, v))| {
            match (arg.kind(), *v) {
                (_, ty::Invariant) => Some((i, arg)),
                (ty::GenericArgKind::Lifetime(_), ty::Bivariant) => None,
                _ => panic!("unexpected opaque type arg variance"),
            }
        })
    }

    pub fn fold_captured_lifetime_args(
        self,
        tcx: I,
        mut f: impl FnMut(I::Region) -> I::Region,
    ) -> Self {
        let Self { def_id, args } = self;
        let variances = tcx.variances_of(def_id.into());
        let args =
            std::iter::zip(args, variances.into_iter()).map(|(arg, v)| match (arg.kind(), *v) {
                (ty::GenericArgKind::Lifetime(_), ty::Bivariant) => arg,
                (ty::GenericArgKind::Lifetime(lt), _) => f(lt).into(),
                _ => arg,
            });
        let args = tcx.mk_args_from_iter(args);
        Self { def_id, args }
    }
}
