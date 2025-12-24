use std::fmt;

use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, HashStable_NoContext};
use rustc_type_ir_macros::{
    GenericTypeVisitable, Lift_Generic, TypeFoldable_Generic, TypeVisitable_Generic,
};

use crate::{self as ty, BoundVarIndexKind, Interner};

/// Represents a constant in Rust.
#[derive_where(Clone, Copy, Hash, PartialEq; I: Interner)]
#[derive(GenericTypeVisitable)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, HashStable_NoContext)
)]
pub enum ConstKind<I: Interner> {
    /// A const generic parameter.
    Param(I::ParamConst),

    /// Infer the value of the const.
    Infer(InferConst),

    /// Bound const variable, used only when preparing a trait query.
    Bound(BoundVarIndexKind, I::BoundConst),

    /// A placeholder const - universally quantified higher-ranked const.
    Placeholder(I::PlaceholderConst),

    /// An unnormalized const item such as an anon const or assoc const or free const item.
    /// Right now anything other than anon consts does not actually work properly but this
    /// should
    Unevaluated(ty::UnevaluatedConst<I>),

    /// Used to hold computed value.
    Value(I::ValueConst),

    /// A placeholder for a const which could not be computed; this is
    /// propagated to avoid useless error messages.
    Error(I::ErrorGuaranteed),

    /// Unevaluated non-const-item, used by `feature(generic_const_exprs)` to represent
    /// const arguments such as `N + 1` or `foo(N)`
    Expr(I::ExprConst),
}

impl<I: Interner> Eq for ConstKind<I> {}

impl<I: Interner> fmt::Debug for ConstKind<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ConstKind::*;

        match self {
            Param(param) => write!(f, "{param:?}"),
            Infer(var) => write!(f, "{var:?}"),
            Bound(debruijn, var) => crate::debug_bound_var(f, *debruijn, var),
            Placeholder(placeholder) => write!(f, "{placeholder:?}"),
            Unevaluated(uv) => write!(f, "{uv:?}"),
            Value(val) => write!(f, "{val:?}"),
            Error(_) => write!(f, "{{const error}}"),
            Expr(expr) => write!(f, "{expr:?}"),
        }
    }
}

/// An unevaluated (potentially generic) constant used in the type-system.
#[derive_where(Clone, Copy, Debug, Hash, PartialEq; I: Interner)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub struct UnevaluatedConst<I: Interner> {
    pub def: I::UnevaluatedConstId,
    pub args: I::GenericArgs,
}

impl<I: Interner> Eq for UnevaluatedConst<I> {}

impl<I: Interner> UnevaluatedConst<I> {
    #[inline]
    pub fn new(def: I::UnevaluatedConstId, args: I::GenericArgs) -> UnevaluatedConst<I> {
        UnevaluatedConst { def, args }
    }
}

rustc_index::newtype_index! {
    /// A **`const`** **v**ariable **ID**.
    #[encodable]
    #[orderable]
    #[debug_format = "?{}c"]
    #[gate_rustc_only]
    pub struct ConstVid {}
}

/// An inference variable for a const, for use in const generics.
#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "nightly", derive(Encodable_NoContext, Decodable_NoContext))]
pub enum InferConst {
    /// Infer the value of the const.
    Var(ConstVid),
    /// A fresh const variable. See `infer::freshen` for more details.
    Fresh(u32),
}

impl fmt::Debug for InferConst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InferConst::Var(var) => write!(f, "{var:?}"),
            InferConst::Fresh(var) => write!(f, "Fresh({var:?})"),
        }
    }
}

#[cfg(feature = "nightly")]
impl<CTX> HashStable<CTX> for InferConst {
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        match self {
            InferConst::Var(_) => {
                panic!("const variables should not be hashed: {self:?}")
            }
            InferConst::Fresh(i) => i.hash_stable(hcx, hasher),
        }
    }
}

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
/// lists of (nested) `ty::Const`s (which may indirectly contain more `ValTree`s).
#[derive_where(Clone, Debug, Hash, Eq, PartialEq; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub enum ValTreeKind<I: Interner> {
    /// integers, `bool`, `char` are represented as scalars.
    /// See the `ScalarInt` documentation for how `ScalarInt` guarantees that equal values
    /// of these types have the same representation.
    Leaf(I::ScalarInt),

    /// The fields of any kind of aggregate. Structs, tuples and arrays are represented by
    /// listing their fields' values in order.
    ///
    /// Enums are represented by storing their variant index as a u32 field, followed by all
    /// the fields of the variant.
    ///
    /// ZST types are represented as an empty slice.
    // FIXME(mgca): Use a `List` here instead of a boxed slice
    Branch(Box<[I::Const]>),
}

impl<I: Interner> ValTreeKind<I> {
    /// Converts to a `ValTreeKind::Leaf` value, `panic`'ing
    /// if this valtree is some other kind.
    #[inline]
    pub fn to_leaf(&self) -> I::ScalarInt {
        match self {
            ValTreeKind::Leaf(s) => *s,
            ValTreeKind::Branch(..) => panic!("expected leaf, got {:?}", self),
        }
    }

    /// Converts to a `ValTreeKind::Branch` value, `panic`'ing
    /// if this valtree is some other kind.
    #[inline]
    pub fn to_branch(&self) -> &[I::Const] {
        match self {
            ValTreeKind::Branch(branch) => &**branch,
            ValTreeKind::Leaf(..) => panic!("expected branch, got {:?}", self),
        }
    }

    /// Attempts to convert to a `ValTreeKind::Leaf` value.
    pub fn try_to_leaf(&self) -> Option<I::ScalarInt> {
        match self {
            ValTreeKind::Leaf(s) => Some(*s),
            ValTreeKind::Branch(_) => None,
        }
    }

    /// Attempts to convert to a `ValTreeKind::Branch` value.
    pub fn try_to_branch(&self) -> Option<&[I::Const]> {
        match self {
            ValTreeKind::Branch(branch) => Some(&**branch),
            ValTreeKind::Leaf(_) => None,
        }
    }
}
