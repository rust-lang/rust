use std::fmt;
use std::ops::Index;

use arrayvec::ArrayVec;
use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, HashStable_NoContext};
use rustc_type_ir_macros::{Lift_Generic, TypeFoldable_Generic, TypeVisitable_Generic};

use crate::data_structures::HashMap;
use crate::inherent::*;
use crate::{self as ty, Interner, TypingMode, UniverseIndex};

#[derive_where(Clone, Hash, PartialEq, Debug; I: Interner, V)]
#[derive_where(Copy; I: Interner, V: Copy)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, HashStable_NoContext)
)]
pub struct CanonicalQueryInput<I: Interner, V> {
    pub canonical: Canonical<I, V>,
    pub typing_mode: TypingMode<I>,
}

impl<I: Interner, V: Eq> Eq for CanonicalQueryInput<I, V> {}

/// A "canonicalized" type `V` is one where all free inference
/// variables have been rewritten to "canonical vars". These are
/// numbered starting from 0 in order of first appearance.
#[derive_where(Clone, Hash, PartialEq, Debug; I: Interner, V)]
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

impl<I: Interner, V: Eq> Eq for Canonical<I, V> {}

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
#[derive_where(Clone, Copy, Hash, PartialEq, Debug; I: Interner)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub enum CanonicalVarKind<I: Interner> {
    /// General type variable `?T` that can be unified with arbitrary types.
    ///
    /// We also store the index of the first type variable which is sub-unified
    /// with this one. If there is no inference variable related to this one,
    /// its `sub_root` just points to itself.
    Ty { ui: UniverseIndex, sub_root: ty::BoundVar },

    /// Integral type variable `?I` (that can only be unified with integral types).
    Int,

    /// Floating-point type variable `?F` (that can only be unified with float types).
    Float,

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

impl<I: Interner> Eq for CanonicalVarKind<I> {}

impl<I: Interner> CanonicalVarKind<I> {
    pub fn universe(self) -> UniverseIndex {
        match self {
            CanonicalVarKind::Ty { ui, sub_root: _ } => ui,
            CanonicalVarKind::Region(ui) => ui,
            CanonicalVarKind::Const(ui) => ui,
            CanonicalVarKind::PlaceholderTy(placeholder) => placeholder.universe(),
            CanonicalVarKind::PlaceholderRegion(placeholder) => placeholder.universe(),
            CanonicalVarKind::PlaceholderConst(placeholder) => placeholder.universe(),
            CanonicalVarKind::Float | CanonicalVarKind::Int => UniverseIndex::ROOT,
        }
    }

    /// Replaces the universe of this canonical variable with `ui`.
    ///
    /// In case this is a float or int variable, this causes an ICE if
    /// the updated universe is not the root.
    pub fn with_updated_universe(self, ui: UniverseIndex) -> CanonicalVarKind<I> {
        match self {
            CanonicalVarKind::Ty { ui: _, sub_root } => CanonicalVarKind::Ty { ui, sub_root },
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
            CanonicalVarKind::Int | CanonicalVarKind::Float => {
                assert_eq!(ui, UniverseIndex::ROOT);
                self
            }
        }
    }

    pub fn is_existential(self) -> bool {
        match self {
            CanonicalVarKind::Ty { .. }
            | CanonicalVarKind::Int
            | CanonicalVarKind::Float
            | CanonicalVarKind::Region(_)
            | CanonicalVarKind::Const(_) => true,
            CanonicalVarKind::PlaceholderTy(_)
            | CanonicalVarKind::PlaceholderRegion(..)
            | CanonicalVarKind::PlaceholderConst(_) => false,
        }
    }

    pub fn is_region(self) -> bool {
        match self {
            CanonicalVarKind::Region(_) | CanonicalVarKind::PlaceholderRegion(_) => true,
            CanonicalVarKind::Ty { .. }
            | CanonicalVarKind::Int
            | CanonicalVarKind::Float
            | CanonicalVarKind::PlaceholderTy(_)
            | CanonicalVarKind::Const(_)
            | CanonicalVarKind::PlaceholderConst(_) => false,
        }
    }

    pub fn expect_placeholder_index(self) -> usize {
        match self {
            CanonicalVarKind::Ty { .. }
            | CanonicalVarKind::Int
            | CanonicalVarKind::Float
            | CanonicalVarKind::Region(_)
            | CanonicalVarKind::Const(_) => {
                panic!("expected placeholder: {self:?}")
            }

            CanonicalVarKind::PlaceholderRegion(placeholder) => placeholder.var().as_usize(),
            CanonicalVarKind::PlaceholderTy(placeholder) => placeholder.var().as_usize(),
            CanonicalVarKind::PlaceholderConst(placeholder) => placeholder.var().as_usize(),
        }
    }
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
#[derive_where(Clone, Copy, Hash, PartialEq, Debug; I: Interner)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, HashStable_NoContext)
)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
pub struct CanonicalVarValues<I: Interner> {
    pub var_values: I::GenericArgs,
}

impl<I: Interner> Eq for CanonicalVarValues<I> {}

impl<I: Interner> CanonicalVarValues<I> {
    pub fn is_identity(&self) -> bool {
        self.var_values.iter().enumerate().all(|(bv, arg)| match arg.kind() {
            ty::GenericArgKind::Lifetime(r) => {
                matches!(r.kind(), ty::ReBound(ty::BoundVarIndexKind::Canonical, br) if br.var().as_usize() == bv)
            }
            ty::GenericArgKind::Type(ty) => {
                matches!(ty.kind(), ty::Bound(ty::BoundVarIndexKind::Canonical, bt) if bt.var().as_usize() == bv)
            }
            ty::GenericArgKind::Const(ct) => {
                matches!(ct.kind(), ty::ConstKind::Bound(ty::BoundVarIndexKind::Canonical, bc) if bc.var().as_usize() == bv)
            }
        })
    }

    pub fn is_identity_modulo_regions(&self) -> bool {
        let mut var = ty::BoundVar::ZERO;
        for arg in self.var_values.iter() {
            match arg.kind() {
                ty::GenericArgKind::Lifetime(r) => {
                    if matches!(r.kind(), ty::ReBound(ty::BoundVarIndexKind::Canonical, br) if var == br.var())
                    {
                        var = var + 1;
                    } else {
                        // It's ok if this region var isn't an identity variable
                    }
                }
                ty::GenericArgKind::Type(ty) => {
                    if matches!(ty.kind(), ty::Bound(ty::BoundVarIndexKind::Canonical, bt) if var == bt.var())
                    {
                        var = var + 1;
                    } else {
                        return false;
                    }
                }
                ty::GenericArgKind::Const(ct) => {
                    if matches!(ct.kind(), ty::ConstKind::Bound(ty::BoundVarIndexKind::Canonical, bc) if var == bc.var())
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
                        CanonicalVarKind::Ty { .. }
                        | CanonicalVarKind::Int
                        | CanonicalVarKind::Float
                        | CanonicalVarKind::PlaceholderTy(_) => {
                            Ty::new_canonical_bound(cx, ty::BoundVar::from_usize(i)).into()
                        }
                        CanonicalVarKind::Region(_) | CanonicalVarKind::PlaceholderRegion(_) => {
                            Region::new_canonical_bound(cx, ty::BoundVar::from_usize(i)).into()
                        }
                        CanonicalVarKind::Const(_) | CanonicalVarKind::PlaceholderConst(_) => {
                            Const::new_canonical_bound(cx, ty::BoundVar::from_usize(i)).into()
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

    pub fn instantiate(
        cx: I,
        variables: I::CanonicalVarKinds,
        mut f: impl FnMut(&[I::GenericArg], CanonicalVarKind<I>) -> I::GenericArg,
    ) -> CanonicalVarValues<I> {
        // Instantiating `CanonicalVarValues` is really hot, but limited to less than
        // 4 most of the time. Avoid creating a `Vec` here.
        if variables.len() <= 4 {
            let mut var_values = ArrayVec::<_, 4>::new();
            for info in variables.iter() {
                var_values.push(f(&var_values, info));
            }
            CanonicalVarValues { var_values: cx.mk_args(&var_values) }
        } else {
            CanonicalVarValues::instantiate_cold(cx, variables, f)
        }
    }

    #[cold]
    fn instantiate_cold(
        cx: I,
        variables: I::CanonicalVarKinds,
        mut f: impl FnMut(&[I::GenericArg], CanonicalVarKind<I>) -> I::GenericArg,
    ) -> CanonicalVarValues<I> {
        let mut var_values = Vec::with_capacity(variables.len());
        for info in variables.iter() {
            var_values.push(f(&var_values, info));
        }
        CanonicalVarValues { var_values: cx.mk_args(&var_values) }
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

#[derive_where(Clone, Debug; I: Interner)]
pub struct CanonicalParamEnvCacheEntry<I: Interner> {
    pub param_env: I::ParamEnv,
    pub variables: Vec<I::GenericArg>,
    pub variable_lookup_table: HashMap<I::GenericArg, usize>,
    pub var_kinds: Vec<CanonicalVarKind<I>>,
}
