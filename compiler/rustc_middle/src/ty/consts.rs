use rustc_macros::extension;
use rustc_type_ir::walk::TypeWalker;
use rustc_type_ir::{self as ir};

use crate::mir::interpret::Scalar;
use crate::ty::{self, Ty, TyCtxt};

mod int;
mod kind;
mod lit;
mod valtree;

pub use int::*;
pub use kind::*;
pub use lit::*;
pub use valtree::*;

pub type ConstKind<'tcx> = ir::ConstKind<TyCtxt<'tcx>>;
pub type AliasConst<'tcx> = ir::AliasConst<TyCtxt<'tcx>>;
pub type AliasConstKind<'tcx> = ir::AliasConstKind<TyCtxt<'tcx>>;
pub type Const<'tcx> = ir::Const<TyCtxt<'tcx>>;

#[cfg(target_pointer_width = "64")]
rustc_data_structures::static_assert_size!(ConstKind<'_>, 32);

#[extension(pub trait ConstExt<'tcx>)]
impl<'tcx> Const<'tcx> {
    /// Creates a constant with the given integer value and interns it.
    #[inline]
    fn from_bits(
        tcx: TyCtxt<'tcx>,
        bits: u128,
        typing_env: ty::TypingEnv<'tcx>,
        ty: Ty<'tcx>,
    ) -> Self {
        let size = tcx
            .layout_of(typing_env.as_query_input(ty))
            .unwrap_or_else(|e| panic!("could not compute layout for {ty:?}: {e:?}"))
            .size;
        let valtree =
            ty::ValTree::from_scalar_int(tcx, ScalarInt::try_from_uint(bits, size).unwrap());
        ty::Const::new_value(tcx, valtree, ty)
    }

    #[inline]
    /// Creates an interned zst constant.
    fn zero_sized(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Self {
        ty::Const::new_value(tcx, ty::ValTree::zst(tcx), ty)
    }

    #[inline]
    fn new_value(tcx: TyCtxt<'tcx>, valtree: ty::ValTree<'tcx>, ty: Ty<'tcx>) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Value(ty::Value { ty, valtree }))
    }

    #[inline]
    /// Creates an interned bool constant.
    fn from_bool(tcx: TyCtxt<'tcx>, v: bool) -> Self {
        Self::from_bits(tcx, v as u128, ty::TypingEnv::fully_monomorphized(), tcx.types.bool)
    }

    #[inline]
    /// Creates an interned usize constant.
    fn from_target_usize(tcx: TyCtxt<'tcx>, n: u64) -> Self {
        Self::from_bits(tcx, n as u128, ty::TypingEnv::fully_monomorphized(), tcx.types.usize)
    }

    /// Panics if `self.kind != ty::ConstKind::Value`.
    fn to_value(self) -> ty::Value<'tcx> {
        match self.kind() {
            ty::ConstKind::Value(cv) => cv,
            _ => bug!("expected ConstKind::Value, got {:?}", self.kind()),
        }
    }

    /// Attempts to convert to a value.
    ///
    /// Note that this does not normalize the constant.
    fn try_to_value(self) -> Option<ty::Value<'tcx>> {
        match self.kind() {
            ty::ConstKind::Value(cv) => Some(cv),
            _ => None,
        }
    }

    /// Converts to a `ValTreeKind::Leaf` value, `panic`'ing
    /// if this constant is some other kind.
    ///
    /// Note that this does not normalize the constant.
    #[inline]
    fn to_leaf(self) -> ScalarInt {
        self.to_value().to_leaf()
    }

    /// Converts to a `ValTreeKind::Branch` value, `panic`'ing
    /// if this constant is some other kind.
    ///
    /// Note that this does not normalize the constant.
    #[inline]
    fn to_branch(self) -> &'tcx [ty::Const<'tcx>] {
        self.to_value().to_branch()
    }

    /// Attempts to convert to a `ValTreeKind::Leaf` value.
    ///
    /// Note that this does not normalize the constant.
    fn try_to_leaf(self) -> Option<ScalarInt> {
        self.try_to_value()?.try_to_leaf()
    }

    /// Attempts to convert to a `ValTreeKind::Leaf` value.
    ///
    /// Note that this does not normalize the constant.
    fn try_to_scalar(self) -> Option<Scalar> {
        self.try_to_leaf().map(Scalar::Int)
    }

    /// Attempts to convert to a `ValTreeKind::Branch` value.
    ///
    /// Note that this does not normalize the constant.
    fn try_to_branch(self) -> Option<&'tcx [ty::Const<'tcx>]> {
        self.try_to_value()?.try_to_branch()
    }

    /// Convenience method to extract the value of a usize constant,
    /// useful to get the length of an array type.
    ///
    /// Note that this does not evaluate the constant.
    #[inline]
    fn try_to_target_usize(self, tcx: TyCtxt<'tcx>) -> Option<u64> {
        self.try_to_value()?.try_to_target_usize(tcx)
    }

    /// Iterator that walks `self` and any types reachable from
    /// `self`, in depth-first order. Note that just walks the types
    /// that appear in `self`, it does not descend into the fields of
    /// structs or variants. For example:
    ///
    /// ```text
    /// isize => { isize }
    /// Foo<Bar<isize>> => { Foo<Bar<isize>>, Bar<isize>, isize }
    /// [isize] => { [isize], isize }
    /// ```
    fn walk(self) -> TypeWalker<TyCtxt<'tcx>> {
        TypeWalker::new(self.into())
    }
}
