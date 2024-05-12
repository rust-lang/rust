#[cfg(feature = "nightly")]
use rustc_macros::{HashStable_NoContext, TyDecodable, TyEncodable};
use rustc_type_ir_macros::{TypeFoldable_Generic, TypeVisitable_Generic};
use std::fmt;
use std::hash::Hash;

use crate::inherent::*;
use crate::{Interner, UniverseIndex};

/// A "canonicalized" type `V` is one where all free inference
/// variables have been rewritten to "canonical vars". These are
/// numbered starting from 0 in order of first appearance.
#[derive(derivative::Derivative)]
#[derivative(
    Clone(bound = "V: Clone"),
    Hash(bound = "V: Hash"),
    PartialEq(bound = "V: PartialEq"),
    Eq(bound = "V: Eq"),
    Debug(bound = "V: fmt::Debug"),
    Copy(bound = "V: Copy")
)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(TyEncodable, TyDecodable, HashStable_NoContext))]
pub struct Canonical<I: Interner, V> {
    pub value: V,
    pub max_universe: UniverseIndex,
    // FIXME(lcnr, oli-obk): try moving this into the query inputs instead
    pub defining_opaque_types: I::DefiningOpaqueTypes,
    pub variables: I::CanonicalVars,
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
        let Canonical { defining_opaque_types, max_universe, variables, value } = self;
        Canonical { defining_opaque_types, max_universe, variables, value: map_op(value) }
    }

    /// Allows you to map the `value` of a canonical while keeping the same set of
    /// bound variables.
    ///
    /// **WARNING:** This function is very easy to mis-use, hence the name! See
    /// the comment of [Canonical::unchecked_map] for more details.
    pub fn unchecked_rebind<W>(self, value: W) -> Canonical<I, W> {
        let Canonical { defining_opaque_types, max_universe, variables, value: _ } = self;
        Canonical { defining_opaque_types, max_universe, variables, value }
    }
}

impl<I: Interner, V: fmt::Display> fmt::Display for Canonical<I, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { value, max_universe, variables, defining_opaque_types } = self;
        write!(
            f,
            "Canonical {{ value: {value}, max_universe: {max_universe:?}, variables: {variables:?}, defining_opaque_types: {defining_opaque_types:?} }}",
        )
    }
}

/// Information about a canonical variable that is included with the
/// canonical value. This is sufficient information for code to create
/// a copy of the canonical value in some other inference context,
/// with fresh inference variables replacing the canonical values.
#[derive(derivative::Derivative)]
#[derivative(
    Clone(bound = ""),
    Copy(bound = ""),
    Hash(bound = ""),
    Debug(bound = ""),
    Eq(bound = ""),
    PartialEq(bound = "")
)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(TyDecodable, TyEncodable, HashStable_NoContext))]
pub struct CanonicalVarInfo<I: Interner> {
    pub kind: CanonicalVarKind<I>,
}

impl<I: Interner> CanonicalVarInfo<I> {
    pub fn universe(self) -> UniverseIndex {
        self.kind.universe()
    }

    #[must_use]
    pub fn with_updated_universe(self, ui: UniverseIndex) -> CanonicalVarInfo<I> {
        CanonicalVarInfo { kind: self.kind.with_updated_universe(ui) }
    }

    pub fn is_existential(&self) -> bool {
        match self.kind {
            CanonicalVarKind::Ty(_) => true,
            CanonicalVarKind::PlaceholderTy(_) => false,
            CanonicalVarKind::Region(_) => true,
            CanonicalVarKind::PlaceholderRegion(..) => false,
            CanonicalVarKind::Const(..) => true,
            CanonicalVarKind::PlaceholderConst(_, _) => false,
            CanonicalVarKind::Effect => true,
        }
    }

    pub fn is_region(&self) -> bool {
        match self.kind {
            CanonicalVarKind::Region(_) | CanonicalVarKind::PlaceholderRegion(_) => true,
            CanonicalVarKind::Ty(_)
            | CanonicalVarKind::PlaceholderTy(_)
            | CanonicalVarKind::Const(_, _)
            | CanonicalVarKind::PlaceholderConst(_, _)
            | CanonicalVarKind::Effect => false,
        }
    }

    pub fn expect_placeholder_index(self) -> usize {
        match self.kind {
            CanonicalVarKind::Ty(_)
            | CanonicalVarKind::Region(_)
            | CanonicalVarKind::Const(_, _)
            | CanonicalVarKind::Effect => panic!("expected placeholder: {self:?}"),

            CanonicalVarKind::PlaceholderRegion(placeholder) => placeholder.var().as_usize(),
            CanonicalVarKind::PlaceholderTy(placeholder) => placeholder.var().as_usize(),
            CanonicalVarKind::PlaceholderConst(placeholder, _) => placeholder.var().as_usize(),
        }
    }
}

/// Describes the "kind" of the canonical variable. This is a "kind"
/// in the type-theory sense of the term -- i.e., a "meta" type system
/// that analyzes type-like values.
#[derive(derivative::Derivative)]
#[derivative(Clone(bound = ""), Copy(bound = ""), Hash(bound = ""), Debug(bound = ""))]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(TyDecodable, TyEncodable, HashStable_NoContext))]
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
    Const(UniverseIndex, I::Ty),

    /// Effect variable `'?E`.
    Effect,

    /// A "placeholder" that represents "any const".
    PlaceholderConst(I::PlaceholderConst, I::Ty),
}

impl<I: Interner> PartialEq for CanonicalVarKind<I> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Ty(l0), Self::Ty(r0)) => l0 == r0,
            (Self::PlaceholderTy(l0), Self::PlaceholderTy(r0)) => l0 == r0,
            (Self::Region(l0), Self::Region(r0)) => l0 == r0,
            (Self::PlaceholderRegion(l0), Self::PlaceholderRegion(r0)) => l0 == r0,
            (Self::Const(l0, l1), Self::Const(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::PlaceholderConst(l0, l1), Self::PlaceholderConst(r0, r1)) => {
                l0 == r0 && l1 == r1
            }
            _ => std::mem::discriminant(self) == std::mem::discriminant(other),
        }
    }
}

impl<I: Interner> CanonicalVarKind<I> {
    pub fn universe(self) -> UniverseIndex {
        match self {
            CanonicalVarKind::Ty(CanonicalTyVarKind::General(ui)) => ui,
            CanonicalVarKind::Region(ui) => ui,
            CanonicalVarKind::Const(ui, _) => ui,
            CanonicalVarKind::PlaceholderTy(placeholder) => placeholder.universe(),
            CanonicalVarKind::PlaceholderRegion(placeholder) => placeholder.universe(),
            CanonicalVarKind::PlaceholderConst(placeholder, _) => placeholder.universe(),
            CanonicalVarKind::Ty(CanonicalTyVarKind::Float | CanonicalTyVarKind::Int) => {
                UniverseIndex::ROOT
            }
            CanonicalVarKind::Effect => UniverseIndex::ROOT,
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
            CanonicalVarKind::Const(_, ty) => CanonicalVarKind::Const(ui, ty),

            CanonicalVarKind::PlaceholderTy(placeholder) => {
                CanonicalVarKind::PlaceholderTy(placeholder.with_updated_universe(ui))
            }
            CanonicalVarKind::PlaceholderRegion(placeholder) => {
                CanonicalVarKind::PlaceholderRegion(placeholder.with_updated_universe(ui))
            }
            CanonicalVarKind::PlaceholderConst(placeholder, ty) => {
                CanonicalVarKind::PlaceholderConst(placeholder.with_updated_universe(ui), ty)
            }
            CanonicalVarKind::Ty(CanonicalTyVarKind::Int | CanonicalTyVarKind::Float)
            | CanonicalVarKind::Effect => {
                assert_eq!(ui, UniverseIndex::ROOT);
                self
            }
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
#[cfg_attr(feature = "nightly", derive(TyDecodable, TyEncodable, HashStable_NoContext))]
pub enum CanonicalTyVarKind {
    /// General type variable `?T` that can be unified with arbitrary types.
    General(UniverseIndex),

    /// Integral type variable `?I` (that can only be unified with integral types).
    Int,

    /// Floating-point type variable `?F` (that can only be unified with float types).
    Float,
}
