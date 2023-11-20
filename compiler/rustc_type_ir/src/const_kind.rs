#[cfg(feature = "nightly")]
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use std::fmt;

use crate::{DebruijnIndex, DebugWithInfcx, InferCtxtLike, Interner, WithInfcx};

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
#[cfg_attr(feature = "nightly", derive(TyEncodable, TyDecodable))]
pub enum ConstKind<I: Interner> {
    /// A const generic parameter.
    Param(I::ParamConst),

    /// Infer the value of the const.
    Infer(InferConst),

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

#[cfg(feature = "nightly")]
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

#[cfg(feature = "nightly")]
impl<CTX: crate::HashStableContext, I: Interner> HashStable<CTX> for ConstKind<I>
where
    I::ParamConst: HashStable<CTX>,
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

rustc_index::newtype_index! {
    /// A **`const`** **v**ariable **ID**.
    #[debug_format = "?{}c"]
    #[gate_rustc_only]
    pub struct ConstVid {}
}

rustc_index::newtype_index! {
    /// An **effect** **v**ariable **ID**.
    ///
    /// Handling effect infer variables happens separately from const infer variables
    /// because we do not want to reuse any of the const infer machinery. If we try to
    /// relate an effect variable with a normal one, we would ICE, which can catch bugs
    /// where we are not correctly using the effect var for an effect param. Fallback
    /// is also implemented on top of having separate effect and normal const variables.
    #[debug_format = "?{}e"]
    #[gate_rustc_only]
    pub struct EffectVid {}
}

/// An inference variable for a const, for use in const generics.
#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "nightly", derive(TyEncodable, TyDecodable))]
pub enum InferConst {
    /// Infer the value of the const.
    Var(ConstVid),
    /// Infer the value of the effect.
    ///
    /// For why this is separate from the `Var` variant above, see the
    /// documentation on `EffectVid`.
    EffectVar(EffectVid),
    /// A fresh const variable. See `infer::freshen` for more details.
    Fresh(u32),
}

impl fmt::Debug for InferConst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InferConst::Var(var) => write!(f, "{var:?}"),
            InferConst::EffectVar(var) => write!(f, "{var:?}"),
            InferConst::Fresh(var) => write!(f, "Fresh({var:?})"),
        }
    }
}
impl<I: Interner> DebugWithInfcx<I> for InferConst {
    fn fmt<Infcx: InferCtxtLike<Interner = I>>(
        this: WithInfcx<'_, Infcx, &Self>,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        match this.infcx.universe_of_ct(*this.data) {
            None => write!(f, "{:?}", this.data),
            Some(universe) => match *this.data {
                InferConst::Var(vid) => write!(f, "?{}_{}c", vid.index(), universe.index()),
                InferConst::EffectVar(vid) => write!(f, "?{}_{}e", vid.index(), universe.index()),
                InferConst::Fresh(_) => {
                    unreachable!()
                }
            },
        }
    }
}

#[cfg(feature = "nightly")]
impl<CTX> HashStable<CTX> for InferConst {
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        match self {
            InferConst::Var(_) | InferConst::EffectVar(_) => {
                panic!("const variables should not be hashed: {self:?}")
            }
            InferConst::Fresh(i) => i.hash_stable(hcx, hasher),
        }
    }
}
