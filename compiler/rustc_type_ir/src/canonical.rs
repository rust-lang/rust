use std::fmt;
use std::hash::Hash;
use std::ops::Index;

use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, HashStable_NoContext};
use rustc_type_ir_macros::{Lift_Generic, TypeFoldable_Generic, TypeVisitable_Generic};

use crate::inherent::*;
use crate::{self as ty, Interner, TypingMode, UniverseIndex};

#[derive_where(Clone; I: Interner, V: Clone)]
#[derive_where(Hash; I: Interner, V: Hash)]
#[derive_where(PartialEq; I: Interner, V: PartialEq)]
#[derive_where(Eq; I: Interner, V: Eq)]
#[derive_where(Debug; I: Interner, V: fmt::Debug)]
#[derive_where(Copy; I: Interner, V: Copy)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, HashStable_NoContext)
)]
pub struct CanonicalQueryInput<I: Interner, V> {
    pub canonical: Canonical<I, V>,
    pub typing_mode: TypingMode<I>,
}

/// A "canonicalized" type `V` is one where all free inference
/// variables have been rewritten to "canonical vars". These are
/// numbered starting from 0 in order of first appearance.
#[derive_where(Clone; I: Interner, V: Clone)]
#[derive_where(Hash; I: Interner, V: Hash)]
#[derive_where(PartialEq; I: Interner, V: PartialEq)]
#[derive_where(Eq; I: Interner, V: Eq)]
#[derive_where(Debug; I: Interner, V: fmt::Debug)]
#[derive_where(Copy; I: Interner, V: Copy)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, HashStable_NoContext)
)]
pub struct Canonical<I: Interner, V> {
    pub value: V,
    pub max_universe: UniverseIndex,
    pub variables: I::CanonicalVarKinds,
}

impl<I: Interner, V> Canonical<I, V> {
    /// Allows you to map the `value` of a canonical while keeping the
    /// same set of bound variables.
    ///
    /// **WARNING:** This function is very easy to mis-use, hence the
    /// name!  In particular, the new value `W` must use all **the
    /// same type/region variables** in **precisely the same order**
    /// as the original! (The ordering is defined by the
    /// `TypeFoldable` implementation of the type in question.)
    ///
    /// An example of a **correct** use of this:
    ///
    /// ```rust,ignore (not real code)
    /// let a: Canonical<I, T> = ...;
    /// let b: Canonical<I, (T,)> = a.unchecked_map(|v| (v, ));
    /// ```
    ///
    /// An example of an **incorrect** use of this:
    ///
    /// ```rust,ignore (not real code)
    /// let a: Canonical<I, T> = ...;
    /// let ty: Ty<I> = ...;
    /// let b: Canonical<I, (T, Ty<I>)> = a.unchecked_map(|v| (v, ty));
    /// ```
    pub fn unchecked_map<W>(self, map_op: impl FnOnce(V) -> W) -> Canonical<I, W> {
        let Canonical { max_universe, variables, value } = self;
        Canonical { max_universe, variables, value: map_op(value) }
    }
}

impl<I: Interner, V: fmt::Display> fmt::Display for Canonical<I, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { value, max_universe, variables } = self;
        write!(
            f,
            "Canonical {{ value: {value}, max_universe: {max_universe:?}, variables: {variables:?} }}",
        )
    }
}

/// Information about a canonical variable that is included with the
/// canonical value. This is sufficient information for code to create
/// a copy of the canonical value in some other inference context,
/// with fresh inference variables replacing the canonical values.
#[derive_where(Clone, Copy, Hash, PartialEq, Eq, Debug; I: Interner)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub enum CanonicalVarKind<I: Interner> {
    /// Some kind of type inference variable.
    Ty(CanonicalTyVarKind),

    /// A "placeholder" that represents "any type".
    PlaceholderTy(I::PlaceholderTy),

    /// Region variable `'?R`.
    Region(UniverseIndex),

    /// A "placeholder" that represents "any region". Created when you
    /// are solving a goal like `for<'a> T: Foo<'a>` to represent the
    /// bound region `'a`.
    PlaceholderRegion(I::PlaceholderRegion),

    /// Some kind of const inference variable.
    Const(UniverseIndex),

    /// A "placeholder" that represents "any const".
    PlaceholderConst(I::PlaceholderConst),
}

impl<I: Interner> CanonicalVarKind<I> {
    pub fn universe(self) -> UniverseIndex {
        match self {
            CanonicalVarKind::Ty(CanonicalTyVarKind::General(ui)) => ui,
            CanonicalVarKind::Region(ui) => ui,
            CanonicalVarKind::Const(ui) => ui,
            CanonicalVarKind::PlaceholderTy(placeholder) => placeholder.universe(),
            CanonicalVarKind::PlaceholderRegion(placeholder) => placeholder.universe(),
            CanonicalVarKind::PlaceholderConst(placeholder) => placeholder.universe(),
            CanonicalVarKind::Ty(CanonicalTyVarKind::Float | CanonicalTyVarKind::Int) => {
                UniverseIndex::ROOT
            }
        }
    }

    /// Replaces the universe of this canonical variable with `ui`.
    ///
    /// In case this is a float or int variable, this causes an ICE if
    /// the updated universe is not the root.
    pub fn with_updated_universe(self, ui: UniverseIndex) -> CanonicalVarKind<I> {
        match self {
            CanonicalVarKind::Ty(CanonicalTyVarKind::General(_)) => {
                CanonicalVarKind::Ty(CanonicalTyVarKind::General(ui))
            }
            CanonicalVarKind::Region(_) => CanonicalVarKind::Region(ui),
            CanonicalVarKind::Const(_) => CanonicalVarKind::Const(ui),

            CanonicalVarKind::PlaceholderTy(placeholder) => {
                CanonicalVarKind::PlaceholderTy(placeholder.with_updated_universe(ui))
            }
            CanonicalVarKind::PlaceholderRegion(placeholder) => {
                CanonicalVarKind::PlaceholderRegion(placeholder.with_updated_universe(ui))
            }
            CanonicalVarKind::PlaceholderConst(placeholder) => {
                CanonicalVarKind::PlaceholderConst(placeholder.with_updated_universe(ui))
            }
            CanonicalVarKind::Ty(CanonicalTyVarKind::Int | CanonicalTyVarKind::Float) => {
                assert_eq!(ui, UniverseIndex::ROOT);
                self
            }
        }
    }

    pub fn is_existential(self) -> bool {
        match self {
            CanonicalVarKind::Ty(_) => true,
            CanonicalVarKind::PlaceholderTy(_) => false,
            CanonicalVarKind::Region(_) => true,
            CanonicalVarKind::PlaceholderRegion(..) => false,
            CanonicalVarKind::Const(_) => true,
            CanonicalVarKind::PlaceholderConst(_) => false,
        }
    }

    pub fn is_region(self) -> bool {
        match self {
            CanonicalVarKind::Region(_) | CanonicalVarKind::PlaceholderRegion(_) => true,
            CanonicalVarKind::Ty(_)
            | CanonicalVarKind::PlaceholderTy(_)
            | CanonicalVarKind::Const(_)
            | CanonicalVarKind::PlaceholderConst(_) => false,
        }
    }

    pub fn expect_placeholder_index(self) -> usize {
        match self {
            CanonicalVarKind::Ty(_) | CanonicalVarKind::Region(_) | CanonicalVarKind::Const(_) => {
                panic!("expected placeholder: {self:?}")
            }

            CanonicalVarKind::PlaceholderRegion(placeholder) => placeholder.var().as_usize(),
            CanonicalVarKind::PlaceholderTy(placeholder) => placeholder.var().as_usize(),
            CanonicalVarKind::PlaceholderConst(placeholder) => placeholder.var().as_usize(),
        }
    }
}

/// Rust actually has more than one category of type variables;
/// notably, the type variables we create for literals (e.g., 22 or
/// 22.) can only be instantiated with integral/float types (e.g.,
/// usize or f32). In order to faithfully reproduce a type, we need to
/// know what set of types a given type variable can be unified with.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub enum CanonicalTyVarKind {
    /// General type variable `?T` that can be unified with arbitrary types.
    General(UniverseIndex),

    /// Integral type variable `?I` (that can only be unified with integral types).
    Int,

    /// Floating-point type variable `?F` (that can only be unified with float types).
    Float,
}

/// A set of values corresponding to the canonical variables from some
/// `Canonical`. You can give these values to
/// `canonical_value.instantiate` to instantiate them into the canonical
/// value at the right places.
///
/// When you canonicalize a value `V`, you get back one of these
/// vectors with the original values that were replaced by canonical
/// variables. You will need to supply it later to instantiate the
/// canonicalized query response.
#[derive_where(Clone, Copy, Hash, PartialEq, Eq, Debug; I: Interner)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, HashStable_NoContext)
)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
pub struct CanonicalVarValues<I: Interner> {
    pub var_values: I::GenericArgs,
}

impl<I: Interner> CanonicalVarValues<I> {
    pub fn is_identity(&self) -> bool {
        self.var_values.iter().enumerate().all(|(bv, arg)| match arg.kind() {
            ty::GenericArgKind::Lifetime(r) => {
                matches!(r.kind(), ty::ReBound(ty::INNERMOST, br) if br.var().as_usize() == bv)
            }
            ty::GenericArgKind::Type(ty) => {
                matches!(ty.kind(), ty::Bound(ty::INNERMOST, bt) if bt.var().as_usize() == bv)
            }
            ty::GenericArgKind::Const(ct) => {
                matches!(ct.kind(), ty::ConstKind::Bound(ty::INNERMOST, bc) if bc.var().as_usize() == bv)
            }
        })
    }

    pub fn is_identity_modulo_regions(&self) -> bool {
        let mut var = ty::BoundVar::ZERO;
        for arg in self.var_values.iter() {
            match arg.kind() {
                ty::GenericArgKind::Lifetime(r) => {
                    if matches!(r.kind(), ty::ReBound(ty::INNERMOST, br) if var == br.var()) {
                        var = var + 1;
                    } else {
                        // It's ok if this region var isn't an identity variable
                    }
                }
                ty::GenericArgKind::Type(ty) => {
                    if matches!(ty.kind(), ty::Bound(ty::INNERMOST, bt) if var == bt.var()) {
                        var = var + 1;
                    } else {
                        return false;
                    }
                }
                ty::GenericArgKind::Const(ct) => {
                    if matches!(ct.kind(), ty::ConstKind::Bound(ty::INNERMOST, bc) if var == bc.var())
                    {
                        var = var + 1;
                    } else {
                        return false;
                    }
                }
            }
        }

        true
    }

    // Given a list of canonical variables, construct a set of values which are
    // the identity response.
    pub fn make_identity(cx: I, infos: I::CanonicalVarKinds) -> CanonicalVarValues<I> {
        CanonicalVarValues {
            var_values: cx.mk_args_from_iter(infos.iter().enumerate().map(
                |(i, kind)| -> I::GenericArg {
                    match kind {
                        CanonicalVarKind::Ty(_) | CanonicalVarKind::PlaceholderTy(_) => {
                            Ty::new_anon_bound(cx, ty::INNERMOST, ty::BoundVar::from_usize(i))
                                .into()
                        }
                        CanonicalVarKind::Region(_) | CanonicalVarKind::PlaceholderRegion(_) => {
                            Region::new_anon_bound(cx, ty::INNERMOST, ty::BoundVar::from_usize(i))
                                .into()
                        }
                        CanonicalVarKind::Const(_) | CanonicalVarKind::PlaceholderConst(_) => {
                            Const::new_anon_bound(cx, ty::INNERMOST, ty::BoundVar::from_usize(i))
                                .into()
                        }
                    }
                },
            )),
        }
    }

    /// Creates dummy var values which should not be used in a
    /// canonical response.
    pub fn dummy() -> CanonicalVarValues<I> {
        CanonicalVarValues { var_values: Default::default() }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.var_values.len()
    }
}

impl<'a, I: Interner> IntoIterator for &'a CanonicalVarValues<I> {
    type Item = I::GenericArg;
    type IntoIter = <I::GenericArgs as SliceLike>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.var_values.iter()
    }
}

impl<I: Interner> Index<ty::BoundVar> for CanonicalVarValues<I> {
    type Output = I::GenericArg;

    fn index(&self, value: ty::BoundVar) -> &I::GenericArg {
        &self.var_values.as_slice()[value.as_usize()]
    }
}
