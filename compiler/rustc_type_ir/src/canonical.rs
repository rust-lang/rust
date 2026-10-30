use std::fmt;
use std::ops::Index;

use arrayvec::ArrayVec;
use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, StableHash_NoContext};
use rustc_type_ir_macros::{
    GenericTypeVisitable, Lift_Generic, TypeFoldable_Generic, TypeVisitable_Generic,
};

use crate::data_structures::{DelayedMap, HashMap};
use crate::inherent::*;
use crate::{self as ty, Interner, TypingModeEqWrapper, UniverseIndex};

#[derive_where(Clone, Hash, PartialEq, Debug; I: Interner, V)]
#[derive_where(Copy; I: Interner, V: Copy)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, StableHash_NoContext)
)]
pub struct CanonicalQueryInput<I: Interner, V> {
    pub canonical: Canonical<I, V>,
    pub typing_mode: TypingModeEqWrapper<I>,
}

impl<I: Interner, V: Eq> Eq for CanonicalQueryInput<I, V> {}

/// A "canonicalized" type `V` is one where all free inference
/// variables have been rewritten to "canonical vars". These are
/// numbered starting from 0 in order of first appearance.
#[derive_where(Clone, Hash, PartialEq, Debug; I: Interner, V)]
#[derive_where(Copy; I: Interner, V: Copy)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, StableHash_NoContext)
)]
pub struct Canonical<I: Interner, V> {
    pub value: V,
    pub max_universe: UniverseIndex,
    pub var_kinds: I::CanonicalVarKinds,
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
        let Canonical { max_universe, var_kinds, value } = self;
        Canonical { max_universe, var_kinds, value: map_op(value) }
    }

    pub fn project<W>(self, f: impl FnOnce(V) -> W) -> Canonicalized<W> {
        Canonicalized(f(self.value))
    }
}

impl<I: Interner, V: fmt::Display> fmt::Display for Canonical<I, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { value, max_universe, var_kinds } = self;
        write!(
            f,
            "Canonical {{ value: {value}, max_universe: {max_universe:?}, var_kinds: {var_kinds:?} }}",
        )
    }
}

/// A projection of a `Canonical<V>`, as produced by `Canonical::project`. Useful as a marker when
/// dealing with fields within a `Canonical<V>`, because it indicates that `T` may contain
/// canonical bound variables that must be instantiated before use.
///
/// The field is private and the contents can only be accessed directly with `into_uninstantiated`,
/// which is easy to spot.
#[derive(Copy, Clone, Debug)]
pub struct Canonicalized<T>(T);

impl<T> Canonicalized<T> {
    pub fn into_uninstantiated(self) -> T {
        self.0
    }
}

impl<T: SliceLike> Canonicalized<T> {
    pub fn len(self) -> usize {
        self.0.len()
    }

    pub fn iter(self) -> impl Iterator<Item = Canonicalized<T::Item>> {
        self.0.iter().map(|x| Canonicalized(x))
    }
}

/// Information about a canonical variable that is included with the
/// canonical value. This is sufficient information for code to create
/// a copy of the canonical value in some other inference context,
/// with fresh inference variables replacing the canonical values.
#[derive_where(Clone, Copy, Hash, PartialEq, Debug; I: Interner)]
#[derive(GenericTypeVisitable)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, StableHash_NoContext)
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
    PlaceholderTy(ty::PlaceholderType<I>),

    /// Region variable `'?R`.
    Region(UniverseIndex),

    /// A "placeholder" that represents "any region". Created when you
    /// are solving a goal like `for<'a> T: Foo<'a>` to represent the
    /// bound region `'a`.
    PlaceholderRegion(ty::PlaceholderRegion<I>),

    /// Some kind of const inference variable.
    Const(UniverseIndex),

    /// A "placeholder" that represents "any const".
    PlaceholderConst(ty::PlaceholderConst<I>),
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
    derive(Encodable_NoContext, Decodable_NoContext, StableHash_NoContext)
)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic, Lift_Generic)]
pub struct CanonicalVarValues<I: Interner> {
    pub var_values: I::GenericArgs,
}

impl<I: Interner> Eq for CanonicalVarValues<I> {}

impl<I: Interner> CanonicalVarValues<I> {
    pub fn instantiate(
        cx: I,
        var_kinds: I::CanonicalVarKinds,
        mut f: impl FnMut(&[I::GenericArg], CanonicalVarKind<I>) -> I::GenericArg,
    ) -> CanonicalVarValues<I> {
        // Instantiating `CanonicalVarValues` is really hot, but limited to less than
        // 4 most of the time. Avoid creating a `Vec` here.
        if var_kinds.len() <= 4 {
            let mut var_values = ArrayVec::<_, 4>::new();
            for info in var_kinds.iter() {
                // Note: the `f` call gets all the previously constructed `var_values`.
                var_values.push(f(&var_values, info));
            }
            CanonicalVarValues { var_values: cx.mk_args(&var_values) }
        } else {
            CanonicalVarValues::instantiate_cold(cx, var_kinds, f)
        }
    }

    #[cold]
    fn instantiate_cold(
        cx: I,
        var_kinds: I::CanonicalVarKinds,
        mut f: impl FnMut(&[I::GenericArg], CanonicalVarKind<I>) -> I::GenericArg,
    ) -> CanonicalVarValues<I> {
        let mut var_values = Vec::with_capacity(var_kinds.len());
        for info in var_kinds.iter() {
            var_values.push(f(&var_values, info));
        }
        CanonicalVarValues { var_values: cx.mk_args(&var_values) }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.var_values.len()
    }
}

impl<I: Interner> Index<ty::BoundVar> for CanonicalVarValues<I> {
    type Output = I::GenericArg;

    fn index(&self, value: ty::BoundVar) -> &I::GenericArg {
        &self.var_values.as_slice()[value.as_usize()]
    }
}

#[derive_where(Default; I: Interner)]
pub struct CanonicalParamEnvCache<I: Interner>(
    pub HashMap<I::ParamEnv, CanonicalParamEnvCacheEntry<I>>,
);

#[derive_where(Clone, Debug; I: Interner)]
pub struct CanonicalParamEnvCacheEntry<I: Interner> {
    // Note: this `param_env` is the canonicalized form of the key for this entry in the enclosing
    // `CanonicalParamEnvCache`.
    pub param_env: I::ParamEnv,
    pub variables: Vec<I::GenericArg>,
    pub variable_lookup_table: HashMap<I::GenericArg, usize>,
    pub var_kinds: Vec<CanonicalVarKind<I>>,
}

/// State used and modified by a canonicalizer during canonicalization. To avoid many allocations,
/// this state is reused by many canonicalizers from `GlobalCaches`.
#[derive_where(Default; I: Interner)]
pub struct CanonicalizerState<I: Interner> {
    pub variables: Vec<I::GenericArg>,
    pub var_kinds: Vec<CanonicalVarKind<I>>,
    pub variable_lookup_table: HashMap<I::GenericArg, usize>,

    /// Maps each `sub_unification_table_root_var` to the index of the first
    /// variable which used it.
    ///
    /// This means in case two type variables have the same sub relations root,
    /// we set the `sub_root` of the second variable to the position of the first.
    /// Otherwise the `sub_root` of each type variable is just its own position.
    pub sub_root_lookup_table: HashMap<ty::TyVid, usize>,

    /// We can simply cache based on the ty itself, because we use
    /// `ty::BoundVarIndexKind::Canonical`.
    pub cache: DelayedMap<I::Ty, I::Ty>,
}

impl<I: Interner> CanonicalizerState<I> {
    pub fn clear(&mut self) {
        // Deconstruct to ensure no fields are missed.
        let Self { variables, var_kinds, variable_lookup_table, sub_root_lookup_table, cache } =
            self;

        // The vectors can be simply cleared because `Vec::clear` is cheap: it just sets the length
        // to zero. (The elements don't impl `Drop`.)
        //
        // The hash maps are different: `HashMap::clear` has to write some control bytes, the
        // number of which is proportional to capacity. Most of the time the hash map capacity is
        // small and it's not a problem, but on some benchmarks it can occasionally get very big.
        // In which case we just start over with a fresh hash map so we aren't saddled with a high
        // `clear` cost forevermore.

        variables.clear();
        var_kinds.clear();
        if variable_lookup_table.capacity() <= 128 {
            variable_lookup_table.clear();
        } else {
            *variable_lookup_table = HashMap::default();
        }
        if sub_root_lookup_table.capacity() <= 128 {
            sub_root_lookup_table.clear();
        } else {
            *sub_root_lookup_table = HashMap::default();
        }
        if cache.capacity() <= 128 {
            cache.clear();
        } else {
            *cache = Default::default();
        }
    }
}
