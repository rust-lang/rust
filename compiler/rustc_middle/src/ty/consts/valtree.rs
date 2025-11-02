use std::fmt;
use std::ops::Deref;

use rustc_data_structures::intern::Interned;
use rustc_hir::def::Namespace;
use rustc_macros::{HashStable, Lift, TyDecodable, TyEncodable, TypeFoldable, TypeVisitable};

use super::ScalarInt;
use crate::mir::interpret::{ErrorHandled, Scalar};
use crate::ty::print::{FmtPrinter, PrettyPrinter};
use crate::ty::{self, Ty, TyCtxt};

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
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
#[derive(HashStable, TyEncodable, TyDecodable)]
pub enum ValTreeKind<'tcx> {
    /// integers, `bool`, `char` are represented as scalars.
    /// See the `ScalarInt` documentation for how `ScalarInt` guarantees that equal values
    /// of these types have the same representation.
    Leaf(ScalarInt),

    //SliceOrStr(ValSlice<'tcx>),
    // don't use SliceOrStr for now
    /// The fields of any kind of aggregate. Structs, tuples and arrays are represented by
    /// listing their fields' values in order.
    ///
    /// Enums are represented by storing their variant index as a u32 field, followed by all
    /// the fields of the variant.
    ///
    /// ZST types are represented as an empty slice.
    Branch(Box<[ValTree<'tcx>]>),
}

impl<'tcx> ValTreeKind<'tcx> {
    #[inline]
    pub fn unwrap_leaf(&self) -> ScalarInt {
        match self {
            Self::Leaf(s) => *s,
            _ => bug!("expected leaf, got {:?}", self),
        }
    }

    #[inline]
    pub fn unwrap_branch(&self) -> &[ValTree<'tcx>] {
        match self {
            Self::Branch(branch) => &**branch,
            _ => bug!("expected branch, got {:?}", self),
        }
    }

    pub fn try_to_scalar(&self) -> Option<Scalar> {
        self.try_to_scalar_int().map(Scalar::Int)
    }

    pub fn try_to_scalar_int(&self) -> Option<ScalarInt> {
        match self {
            Self::Leaf(s) => Some(*s),
            Self::Branch(_) => None,
        }
    }

    pub fn try_to_branch(&self) -> Option<&[ValTree<'tcx>]> {
        match self {
            Self::Branch(branch) => Some(&**branch),
            Self::Leaf(_) => None,
        }
    }
}

/// An interned valtree. Use this rather than `ValTreeKind`, whenever possible.
///
/// See the docs of [`ValTreeKind`] or the [dev guide] for an explanation of this type.
///
/// [dev guide]: https://rustc-dev-guide.rust-lang.org/mir/index.html#valtrees
#[derive(Copy, Clone, Hash, Eq, PartialEq)]
#[derive(HashStable)]
pub struct ValTree<'tcx>(pub(crate) Interned<'tcx, ValTreeKind<'tcx>>);

impl<'tcx> ValTree<'tcx> {
    /// Returns the zero-sized valtree: `Branch([])`.
    pub fn zst(tcx: TyCtxt<'tcx>) -> Self {
        tcx.consts.valtree_zst
    }

    pub fn is_zst(self) -> bool {
        matches!(*self, ValTreeKind::Branch(box []))
    }

    pub fn from_raw_bytes(tcx: TyCtxt<'tcx>, bytes: &[u8]) -> Self {
        let branches = bytes.iter().map(|&b| Self::from_scalar_int(tcx, b.into()));
        Self::from_branches(tcx, branches)
    }

    pub fn from_branches(tcx: TyCtxt<'tcx>, branches: impl IntoIterator<Item = Self>) -> Self {
        tcx.intern_valtree(ValTreeKind::Branch(branches.into_iter().collect()))
    }

    pub fn from_scalar_int(tcx: TyCtxt<'tcx>, i: ScalarInt) -> Self {
        tcx.intern_valtree(ValTreeKind::Leaf(i))
    }
}

impl<'tcx> Deref for ValTree<'tcx> {
    type Target = &'tcx ValTreeKind<'tcx>;

    #[inline]
    fn deref(&self) -> &&'tcx ValTreeKind<'tcx> {
        &self.0.0
    }
}

impl fmt::Debug for ValTree<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

/// `Ok(Err(ty))` indicates the constant was fine, but the valtree couldn't be constructed
/// because the value contains something of type `ty` that is not valtree-compatible.
/// The caller can then show an appropriate error; the query does not have the
/// necessary context to give good user-facing errors for this case.
pub type ConstToValTreeResult<'tcx> = Result<Result<ValTree<'tcx>, Ty<'tcx>>, ErrorHandled>;

/// A type-level constant value.
///
/// Represents a typed, fully evaluated constant.
/// Note that this is also used by pattern elaboration to represent values which cannot occur in types,
/// such as raw pointers and floats.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[derive(HashStable, TyEncodable, TyDecodable, TypeFoldable, TypeVisitable, Lift)]
pub struct Value<'tcx> {
    pub ty: Ty<'tcx>,
    pub valtree: ValTree<'tcx>,
}

impl<'tcx> Value<'tcx> {
    /// Attempts to extract the raw bits from the constant.
    ///
    /// Fails if the value can't be represented as bits (e.g. because it is a reference
    /// or an aggregate).
    #[inline]
    pub fn try_to_bits(self, tcx: TyCtxt<'tcx>, typing_env: ty::TypingEnv<'tcx>) -> Option<u128> {
        let (ty::Bool | ty::Char | ty::Uint(_) | ty::Int(_) | ty::Float(_)) = self.ty.kind() else {
            return None;
        };
        let scalar = self.valtree.try_to_scalar_int()?;
        let input = typing_env.with_post_analysis_normalized(tcx).as_query_input(self.ty);
        let size = tcx.layout_of(input).ok()?.size;
        Some(scalar.to_bits(size))
    }

    pub fn try_to_bool(self) -> Option<bool> {
        if !self.ty.is_bool() {
            return None;
        }
        self.valtree.try_to_scalar_int()?.try_to_bool().ok()
    }

    pub fn try_to_target_usize(self, tcx: TyCtxt<'tcx>) -> Option<u64> {
        if !self.ty.is_usize() {
            return None;
        }
        self.valtree.try_to_scalar_int().map(|s| s.to_target_usize(tcx))
    }

    /// Get the values inside the ValTree as a slice of bytes. This only works for
    /// constants with types &str, &[u8], or [u8; _].
    pub fn try_to_raw_bytes(self, tcx: TyCtxt<'tcx>) -> Option<&'tcx [u8]> {
        match self.ty.kind() {
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
            self.valtree.unwrap_branch().into_iter().map(|v| v.unwrap_leaf().to_u8()),
        ))
    }
}

impl<'tcx> rustc_type_ir::inherent::ValueConst<TyCtxt<'tcx>> for Value<'tcx> {
    fn ty(self) -> Ty<'tcx> {
        self.ty
    }

    fn valtree(self) -> ValTree<'tcx> {
        self.valtree
    }
}

impl<'tcx> fmt::Display for Value<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ty::tls::with(move |tcx| {
            let cv = tcx.lift(*self).unwrap();
            let mut p = FmtPrinter::new(tcx, Namespace::ValueNS);
            p.pretty_print_const_valtree(cv, /*print_ty*/ true)?;
            f.write_str(&p.into_buffer())
        })
    }
}
