use crate::ty::{self, Ty};

pub type PolyCoercePredicate<'tcx> = ty::Binder<'tcx, CoercePredicate<'tcx>>;

/// Encodes that we have to coerce *from* the `a` type to the `b` type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct CoercePredicate<'tcx> {
    pub a: Ty<'tcx>,
    pub b: Ty<'tcx>,
}
