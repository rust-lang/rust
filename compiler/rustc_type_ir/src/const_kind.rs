use rustc_data_structures::stable_hasher::HashStable;
use rustc_data_structures::stable_hasher::StableHasher;
use std::fmt;

use crate::{DebruijnIndex, DebugWithInfcx, HashStableContext, InferCtxtLike, Interner, WithInfcx};

use self::ConstKind::*;

/// Represents a constant in Rust.
#[derive(derivative::Derivative)]
#[derivative(
    Clone(bound = ""),
    PartialOrd(bound = ""),
    PartialOrd = "feature_allow_slow_enum",
    Ord(bound = ""),
    Ord = "feature_allow_slow_enum",
    Hash(bound = "")
)]
#[derive(TyEncodable, TyDecodable)]
pub enum ConstKind<I: Interner> {
    /// A const generic parameter.
    Param(I::ParamConst),

    /// Infer the value of the const.
    Infer(I::InferConst),

    /// Bound const variable, used only when preparing a trait query.
    Bound(DebruijnIndex, I::BoundConst),

    /// A placeholder const - universally quantified higher-ranked const.
    Placeholder(I::PlaceholderConst),

    /// An unnormalized const item such as an anon const or assoc const or free const item.
    /// Right now anything other than anon consts does not actually work properly but this
    /// should
    Unevaluated(I::AliasConst),

    /// Used to hold computed value.
    Value(I::ValueConst),

    /// A placeholder for a const which could not be computed; this is
    /// propagated to avoid useless error messages.
    Error(I::ErrorGuaranteed),

    /// Unevaluated non-const-item, used by `feature(generic_const_exprs)` to represent
    /// const arguments such as `N + 1` or `foo(N)`
    Expr(I::ExprConst),
}

const fn const_kind_discriminant<I: Interner>(value: &ConstKind<I>) -> usize {
    match value {
        Param(_) => 0,
        Infer(_) => 1,
        Bound(_, _) => 2,
        Placeholder(_) => 3,
        Unevaluated(_) => 4,
        Value(_) => 5,
        Error(_) => 6,
        Expr(_) => 7,
    }
}

impl<CTX: HashStableContext, I: Interner> HashStable<CTX> for ConstKind<I>
where
    I::ParamConst: HashStable<CTX>,
    I::InferConst: HashStable<CTX>,
    I::BoundConst: HashStable<CTX>,
    I::PlaceholderConst: HashStable<CTX>,
    I::AliasConst: HashStable<CTX>,
    I::ValueConst: HashStable<CTX>,
    I::ErrorGuaranteed: HashStable<CTX>,
    I::ExprConst: HashStable<CTX>,
{
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        const_kind_discriminant(self).hash_stable(hcx, hasher);
        match self {
            Param(p) => p.hash_stable(hcx, hasher),
            Infer(i) => i.hash_stable(hcx, hasher),
            Bound(d, b) => {
                d.hash_stable(hcx, hasher);
                b.hash_stable(hcx, hasher);
            }
            Placeholder(p) => p.hash_stable(hcx, hasher),
            Unevaluated(u) => u.hash_stable(hcx, hasher),
            Value(v) => v.hash_stable(hcx, hasher),
            Error(e) => e.hash_stable(hcx, hasher),
            Expr(e) => e.hash_stable(hcx, hasher),
        }
    }
}

impl<I: Interner> PartialEq for ConstKind<I> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Param(l0), Param(r0)) => l0 == r0,
            (Infer(l0), Infer(r0)) => l0 == r0,
            (Bound(l0, l1), Bound(r0, r1)) => l0 == r0 && l1 == r1,
            (Placeholder(l0), Placeholder(r0)) => l0 == r0,
            (Unevaluated(l0), Unevaluated(r0)) => l0 == r0,
            (Value(l0), Value(r0)) => l0 == r0,
            (Error(l0), Error(r0)) => l0 == r0,
            (Expr(l0), Expr(r0)) => l0 == r0,
            _ => false,
        }
    }
}

impl<I: Interner> Eq for ConstKind<I> {}

impl<I: Interner> fmt::Debug for ConstKind<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        WithInfcx::with_no_infcx(self).fmt(f)
    }
}

impl<I: Interner> DebugWithInfcx<I> for ConstKind<I> {
    fn fmt<Infcx: InferCtxtLike<Interner = I>>(
        this: WithInfcx<'_, Infcx, &Self>,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        use ConstKind::*;

        match this.data {
            Param(param) => write!(f, "{param:?}"),
            Infer(var) => write!(f, "{:?}", &this.wrap(var)),
            Bound(debruijn, var) => crate::debug_bound_var(f, *debruijn, var.clone()),
            Placeholder(placeholder) => write!(f, "{placeholder:?}"),
            Unevaluated(uv) => {
                write!(f, "{:?}", &this.wrap(uv))
            }
            Value(valtree) => write!(f, "{valtree:?}"),
            Error(_) => write!(f, "{{const error}}"),
            Expr(expr) => write!(f, "{:?}", &this.wrap(expr)),
        }
    }
}
