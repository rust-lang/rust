use super::ScalarInt;
use crate::mir::interpret::{AllocId, Scalar};
use crate::ty::{self, Ty, TyCtxt};
use rustc_macros::{HashStable, TyDecodable, TyEncodable};

#[derive(Copy, Clone, Debug, Hash, TyEncodable, TyDecodable, Eq, PartialEq, Ord, PartialOrd)]
#[derive(HashStable)]
/// This datastructure is used to represent the value of constants used in the type system.
///
/// We explicitly choose a different datastructure from the way values are processed within
/// CTFE, as in the type system equal values (according to their `PartialEq`) must also have
/// equal representation (`==` on the rustc data structure, e.g. `ValTree`) and vice versa.
/// Since CTFE uses `AllocId` to represent pointers, it often happens that two different
/// `AllocId`s point to equal values. So we may end up with different representations for
/// two constants whose value is `&42`. Furthermore any kind of struct that has padding will
/// have arbitrary values within that padding, even if the values of the struct are the same.
///
/// `ValTree` does not have this problem with representation, as it only contains integers or
/// lists of (nested) `ValTree`.
pub enum ValTree<'tcx> {
    /// integers, `bool`, `char` are represented as scalars.
    /// See the `ScalarInt` documentation for how `ScalarInt` guarantees that equal values
    /// of these types have the same representation.
    Leaf(ScalarInt),

    //SliceOrStr(ValSlice<'tcx>),
    // dont use SliceOrStr for now
    /// The fields of any kind of aggregate. Structs, tuples and arrays are represented by
    /// listing their fields' values in order.
    ///
    /// Enums are represented by storing their discriminant as a field, followed by all
    /// the fields of the variant.
    ///
    /// ZST types are represented as an empty slice.
    Branch(&'tcx [ValTree<'tcx>]),
}

impl<'tcx> ValTree<'tcx> {
    pub fn zst() -> Self {
        Self::Branch(&[])
    }

    #[inline]
    pub fn unwrap_leaf(self) -> ScalarInt {
        match self {
            Self::Leaf(s) => s,
            _ => bug!("expected leaf, got {:?}", self),
        }
    }

    #[inline]
    pub fn unwrap_branch(self) -> &'tcx [Self] {
        match self {
            Self::Branch(branch) => branch,
            _ => bug!("expected branch, got {:?}", self),
        }
    }

    pub fn from_raw_bytes<'a>(tcx: TyCtxt<'tcx>, bytes: &'a [u8]) -> Self {
        let branches = bytes.iter().map(|b| Self::Leaf(ScalarInt::from(*b)));
        let interned = tcx.arena.alloc_from_iter(branches);

        Self::Branch(interned)
    }

    pub fn from_scalar_int(i: ScalarInt) -> Self {
        Self::Leaf(i)
    }

    pub fn try_to_scalar(self) -> Option<Scalar<AllocId>> {
        self.try_to_scalar_int().map(Scalar::Int)
    }

    pub fn try_to_scalar_int(self) -> Option<ScalarInt> {
        match self {
            Self::Leaf(s) => Some(s),
            Self::Branch(_) => None,
        }
    }

    pub fn try_to_target_usize(self, tcx: TyCtxt<'tcx>) -> Option<u64> {
        self.try_to_scalar_int().map(|s| s.try_to_target_usize(tcx).ok()).flatten()
    }

    /// Get the values inside the ValTree as a slice of bytes. This only works for
    /// constants with types &str, &[u8], or [u8; _].
    pub fn try_to_raw_bytes(self, tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<&'tcx [u8]> {
        match ty.kind() {
            ty::Ref(_, inner_ty, _) => match inner_ty.kind() {
                // `&str` can be interpreted as raw bytes
                ty::Str => {}
                // `&[u8]` can be interpreted as raw bytes
                ty::Slice(slice_ty) if *slice_ty == tcx.types.u8 => {}
                // other `&_` can't be interpreted as raw bytes
                _ => return None,
            },
            // `[u8; N]` can be interpreted as raw bytes
            ty::Array(array_ty, _) if *array_ty == tcx.types.u8 => {}
            // Otherwise, type cannot be interpreted as raw bytes
            _ => return None,
        }

        Some(tcx.arena.alloc_from_iter(
            self.unwrap_branch().into_iter().map(|v| v.unwrap_leaf().try_to_u8().unwrap()),
        ))
    }
}
