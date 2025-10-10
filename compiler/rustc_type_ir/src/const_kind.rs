use std::fmt;

use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, HashStable_NoContext};
use rustc_type_ir_macros::{Lift_Generic, TypeFoldable_Generic, TypeVisitable_Generic};

use crate::{self as ty, BoundVarIndexKind, Interner};

/// Represents a constant in Rust.
#[derive_where(Clone, Copy, Hash, PartialEq; I: Interner)]
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
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub struct UnevaluatedConst<I: Interner> {
    pub def: I::DefId,
    pub args: I::GenericArgs,
}

impl<I: Interner> Eq for UnevaluatedConst<I> {}

impl<I: Interner> UnevaluatedConst<I> {
    #[inline]
    pub fn new(def: I::DefId, args: I::GenericArgs) -> UnevaluatedConst<I> {
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
