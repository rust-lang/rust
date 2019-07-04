//! **Canonicalization** is the key to constructing a query in the
//! middle of type inference. Ordinarily, it is not possible to store
//! types from type inference in query keys, because they contain
//! references to inference variables whose lifetimes are too short
//! and so forth. Canonicalizing a value T1 using `canonicalize_query`
//! produces two things:
//!
//! - a value T2 where each unbound inference variable has been
//!   replaced with a **canonical variable**;
//! - a map M (of type `CanonicalVarValues`) from those canonical
//!   variables back to the original.
//!
//! We can then do queries using T2. These will give back constraints
//! on the canonical variables which can be translated, using the map
//! M, into constraints in our source context. This process of
//! translating the results back is done by the
//! `instantiate_query_result` method.
//!
//! For a more detailed look at what is happening here, check
//! out the [chapter in the rustc guide][c].
//!
//! [c]: https://rust-lang.github.io/rustc-guide/traits/canonicalization.html

use crate::infer::{InferCtxt, RegionVariableOrigin, TypeVariableOrigin, TypeVariableOriginKind};
use crate::infer::{ConstVariableOrigin, ConstVariableOriginKind};
use crate::infer::region_constraints::MemberConstraint;
use crate::mir::interpret::ConstValue;
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_macros::HashStable;
use serialize::UseSpecializedDecodable;
use smallvec::SmallVec;
use std::ops::Index;
use syntax::source_map::Span;
use crate::ty::fold::TypeFoldable;
use crate::ty::subst::Kind;
use crate::ty::{self, BoundVar, InferConst, Lift, List, Region, TyCtxt};

mod canonicalizer;

pub mod query_response;

mod substitute;

/// A "canonicalized" type `V` is one where all free inference
/// variables have been rewritten to "canonical vars". These are
/// numbered starting from 0 in order of first appearance.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, RustcDecodable, RustcEncodable, HashStable)]
pub struct Canonical<'tcx, V> {
    pub max_universe: ty::UniverseIndex,
    pub variables: CanonicalVarInfos<'tcx>,
    pub value: V,
}

pub type CanonicalVarInfos<'tcx> = &'tcx List<CanonicalVarInfo>;

impl<'tcx> UseSpecializedDecodable for CanonicalVarInfos<'tcx> {}

/// A set of values corresponding to the canonical variables from some
/// `Canonical`. You can give these values to
/// `canonical_value.substitute` to substitute them into the canonical
/// value at the right places.
///
/// When you canonicalize a value `V`, you get back one of these
/// vectors with the original values that were replaced by canonical
/// variables. You will need to supply it later to instantiate the
/// canonicalized query response.
#[derive(Clone, Debug, PartialEq, Eq, Hash, RustcDecodable, RustcEncodable, HashStable)]
pub struct CanonicalVarValues<'tcx> {
    pub var_values: IndexVec<BoundVar, Kind<'tcx>>,
}

/// When we canonicalize a value to form a query, we wind up replacing
/// various parts of it with canonical variables. This struct stores
/// those replaced bits to remember for when we process the query
/// result.
#[derive(Clone, Debug, PartialEq, Eq, Hash, RustcDecodable, RustcEncodable)]
pub struct OriginalQueryValues<'tcx> {
    /// Map from the universes that appear in the query to the
    /// universes in the caller context. For the time being, we only
    /// ever put ROOT values into the query, so this map is very
    /// simple.
    pub universe_map: SmallVec<[ty::UniverseIndex; 4]>,

    /// This is equivalent to `CanonicalVarValues`, but using a
    /// `SmallVec` yields a significant performance win.
    pub var_values: SmallVec<[Kind<'tcx>; 8]>,
}

impl Default for OriginalQueryValues<'tcx> {
    fn default() -> Self {
        let mut universe_map = SmallVec::default();
        universe_map.push(ty::UniverseIndex::ROOT);

        Self {
            universe_map,
            var_values: SmallVec::default(),
        }
    }
}

/// Information about a canonical variable that is included with the
/// canonical value. This is sufficient information for code to create
/// a copy of the canonical value in some other inference context,
/// with fresh inference variables replacing the canonical values.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, RustcDecodable, RustcEncodable, HashStable)]
pub struct CanonicalVarInfo {
    pub kind: CanonicalVarKind,
}

impl CanonicalVarInfo {
    pub fn universe(&self) -> ty::UniverseIndex {
        self.kind.universe()
    }

    pub fn is_existential(&self) -> bool {
        match self.kind {
            CanonicalVarKind::Ty(_) => true,
            CanonicalVarKind::PlaceholderTy(_) => false,
            CanonicalVarKind::Region(_) => true,
            CanonicalVarKind::PlaceholderRegion(..) => false,
            CanonicalVarKind::Const(_) => true,
            CanonicalVarKind::PlaceholderConst(_) => false,
        }
    }
}

/// Describes the "kind" of the canonical variable. This is a "kind"
/// in the type-theory sense of the term -- i.e., a "meta" type system
/// that analyzes type-like values.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, RustcDecodable, RustcEncodable, HashStable)]
pub enum CanonicalVarKind {
    /// Some kind of type inference variable.
    Ty(CanonicalTyVarKind),

    /// A "placeholder" that represents "any type".
    PlaceholderTy(ty::PlaceholderType),

    /// Region variable `'?R`.
    Region(ty::UniverseIndex),

    /// A "placeholder" that represents "any region". Created when you
    /// are solving a goal like `for<'a> T: Foo<'a>` to represent the
    /// bound region `'a`.
    PlaceholderRegion(ty::PlaceholderRegion),

    /// Some kind of const inference variable.
    Const(ty::UniverseIndex),

    /// A "placeholder" that represents "any const".
    PlaceholderConst(ty::PlaceholderConst),
}

impl CanonicalVarKind {
    pub fn universe(self) -> ty::UniverseIndex {
        match self {
            CanonicalVarKind::Ty(kind) => match kind {
                CanonicalTyVarKind::General(ui) => ui,
                CanonicalTyVarKind::Float | CanonicalTyVarKind::Int => ty::UniverseIndex::ROOT,
            }

            CanonicalVarKind::PlaceholderTy(placeholder) => placeholder.universe,
            CanonicalVarKind::Region(ui) => ui,
            CanonicalVarKind::PlaceholderRegion(placeholder) => placeholder.universe,
            CanonicalVarKind::Const(ui) => ui,
            CanonicalVarKind::PlaceholderConst(placeholder) => placeholder.universe,
        }
    }
}

/// Rust actually has more than one category of type variables;
/// notably, the type variables we create for literals (e.g., 22 or
/// 22.) can only be instantiated with integral/float types (e.g.,
/// usize or f32). In order to faithfully reproduce a type, we need to
/// know what set of types a given type variable can be unified with.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, RustcDecodable, RustcEncodable, HashStable)]
pub enum CanonicalTyVarKind {
    /// General type variable `?T` that can be unified with arbitrary types.
    General(ty::UniverseIndex),

    /// Integral type variable `?I` (that can only be unified with integral types).
    Int,

    /// Floating-point type variable `?F` (that can only be unified with float types).
    Float,
}

/// After we execute a query with a canonicalized key, we get back a
/// `Canonical<QueryResponse<..>>`. You can use
/// `instantiate_query_result` to access the data in this result.
#[derive(Clone, Debug, HashStable)]
pub struct QueryResponse<'tcx, R> {
    pub var_values: CanonicalVarValues<'tcx>,
    pub region_constraints: QueryRegionConstraints<'tcx>,
    pub certainty: Certainty,
    pub value: R,
}

#[derive(Clone, Debug, Default, HashStable)]
pub struct QueryRegionConstraints<'tcx> {
    pub outlives: Vec<QueryOutlivesConstraint<'tcx>>,
    pub member_constraints: Vec<MemberConstraint<'tcx>>,
}

impl QueryRegionConstraints<'_> {
    /// Represents an empty (trivially true) set of region
    /// constraints.
    pub fn is_empty(&self) -> bool {
        self.outlives.is_empty() && self.member_constraints.is_empty()
    }
}

pub type Canonicalized<'tcx, V> = Canonical<'tcx, V>;

pub type CanonicalizedQueryResponse<'tcx, T> =
    &'tcx Canonical<'tcx, QueryResponse<'tcx, T>>;

/// Indicates whether or not we were able to prove the query to be
/// true.
#[derive(Copy, Clone, Debug, HashStable)]
pub enum Certainty {
    /// The query is known to be true, presuming that you apply the
    /// given `var_values` and the region-constraints are satisfied.
    Proven,

    /// The query is not known to be true, but also not known to be
    /// false. The `var_values` represent *either* values that must
    /// hold in order for the query to be true, or helpful tips that
    /// *might* make it true. Currently rustc's trait solver cannot
    /// distinguish the two (e.g., due to our preference for where
    /// clauses over impls).
    ///
    /// After some unifiations and things have been done, it makes
    /// sense to try and prove again -- of course, at that point, the
    /// canonical form will be different, making this a distinct
    /// query.
    Ambiguous,
}

impl Certainty {
    pub fn is_proven(&self) -> bool {
        match self {
            Certainty::Proven => true,
            Certainty::Ambiguous => false,
        }
    }

    pub fn is_ambiguous(&self) -> bool {
        !self.is_proven()
    }
}

impl<'tcx, R> QueryResponse<'tcx, R> {
    pub fn is_proven(&self) -> bool {
        self.certainty.is_proven()
    }

    pub fn is_ambiguous(&self) -> bool {
        !self.is_proven()
    }
}

impl<'tcx, R> Canonical<'tcx, QueryResponse<'tcx, R>> {
    pub fn is_proven(&self) -> bool {
        self.value.is_proven()
    }

    pub fn is_ambiguous(&self) -> bool {
        !self.is_proven()
    }
}

impl<'tcx, V> Canonical<'tcx, V> {
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
    /// let a: Canonical<'_, T> = ...;
    /// let b: Canonical<'_, (T,)> = a.unchecked_map(|v| (v, ));
    /// ```
    ///
    /// An example of an **incorrect** use of this:
    ///
    /// ```rust,ignore (not real code)
    /// let a: Canonical<'tcx, T> = ...;
    /// let ty: Ty<'tcx> = ...;
    /// let b: Canonical<'tcx, (T, Ty<'tcx>)> = a.unchecked_map(|v| (v, ty));
    /// ```
    pub fn unchecked_map<W>(self, map_op: impl FnOnce(V) -> W) -> Canonical<'tcx, W> {
        let Canonical {
            max_universe,
            variables,
            value,
        } = self;
        Canonical {
            max_universe,
            variables,
            value: map_op(value),
        }
    }
}

pub type QueryOutlivesConstraint<'tcx> =
    ty::Binder<ty::OutlivesPredicate<Kind<'tcx>, Region<'tcx>>>;

impl<'cx, 'tcx> InferCtxt<'cx, 'tcx> {
    /// Creates a substitution S for the canonical value with fresh
    /// inference variables and applies it to the canonical value.
    /// Returns both the instantiated result *and* the substitution S.
    ///
    /// This is only meant to be invoked as part of constructing an
    /// inference context at the start of a query (see
    /// `InferCtxtBuilder::enter_with_canonical`). It basically
    /// brings the canonical value "into scope" within your new infcx.
    ///
    /// At the end of processing, the substitution S (once
    /// canonicalized) then represents the values that you computed
    /// for each of the canonical inputs to your query.

    pub fn instantiate_canonical_with_fresh_inference_vars<T>(
        &self,
        span: Span,
        canonical: &Canonical<'tcx, T>,
    ) -> (T, CanonicalVarValues<'tcx>)
    where
        T: TypeFoldable<'tcx>,
    {
        // For each universe that is referred to in the incoming
        // query, create a universe in our local inference context. In
        // practice, as of this writing, all queries have no universes
        // in them, so this code has no effect, but it is looking
        // forward to the day when we *do* want to carry universes
        // through into queries.
        let universes: IndexVec<ty::UniverseIndex, _> = std::iter::once(ty::UniverseIndex::ROOT)
            .chain((0..canonical.max_universe.as_u32()).map(|_| self.create_next_universe()))
            .collect();

        let canonical_inference_vars =
            self.instantiate_canonical_vars(span, canonical.variables, |ui| universes[ui]);
        let result = canonical.substitute(self.tcx, &canonical_inference_vars);
        (result, canonical_inference_vars)
    }

    /// Given the "infos" about the canonical variables from some
    /// canonical, creates fresh variables with the same
    /// characteristics (see `instantiate_canonical_var` for
    /// details). You can then use `substitute` to instantiate the
    /// canonical variable with these inference variables.
    fn instantiate_canonical_vars(
        &self,
        span: Span,
        variables: &List<CanonicalVarInfo>,
        universe_map: impl Fn(ty::UniverseIndex) -> ty::UniverseIndex,
    ) -> CanonicalVarValues<'tcx> {
        let var_values: IndexVec<BoundVar, Kind<'tcx>> = variables
            .iter()
            .map(|info| self.instantiate_canonical_var(span, *info, &universe_map))
            .collect();

        CanonicalVarValues { var_values }
    }

    /// Given the "info" about a canonical variable, creates a fresh
    /// variable for it. If this is an existentially quantified
    /// variable, then you'll get a new inference variable; if it is a
    /// universally quantified variable, you get a placeholder.
    fn instantiate_canonical_var(
        &self,
        span: Span,
        cv_info: CanonicalVarInfo,
        universe_map: impl Fn(ty::UniverseIndex) -> ty::UniverseIndex,
    ) -> Kind<'tcx> {
        match cv_info.kind {
            CanonicalVarKind::Ty(ty_kind) => {
                let ty = match ty_kind {
                    CanonicalTyVarKind::General(ui) => {
                        self.next_ty_var_in_universe(
                            TypeVariableOrigin {
                                kind: TypeVariableOriginKind::MiscVariable,
                                span,
                            },
                            universe_map(ui)
                        )
                    }

                    CanonicalTyVarKind::Int => self.next_int_var(),

                    CanonicalTyVarKind::Float => self.next_float_var(),
                };
                ty.into()
            }

            CanonicalVarKind::PlaceholderTy(ty::PlaceholderType { universe, name }) => {
                let universe_mapped = universe_map(universe);
                let placeholder_mapped = ty::PlaceholderType {
                    universe: universe_mapped,
                    name,
                };
                self.tcx.mk_ty(ty::Placeholder(placeholder_mapped)).into()
            }

            CanonicalVarKind::Region(ui) => self.next_region_var_in_universe(
                RegionVariableOrigin::MiscVariable(span),
                universe_map(ui),
            ).into(),

            CanonicalVarKind::PlaceholderRegion(ty::PlaceholderRegion { universe, name }) => {
                let universe_mapped = universe_map(universe);
                let placeholder_mapped = ty::PlaceholderRegion {
                    universe: universe_mapped,
                    name,
                };
                self.tcx.mk_region(ty::RePlaceholder(placeholder_mapped)).into()
            }

            CanonicalVarKind::Const(ui) => {
                self.next_const_var_in_universe(
                    self.next_ty_var_in_universe(
                        TypeVariableOrigin {
                            kind: TypeVariableOriginKind::MiscVariable,
                            span,
                        },
                        universe_map(ui),
                    ),
                    ConstVariableOrigin {
                        kind: ConstVariableOriginKind::MiscVariable,
                        span,
                    },
                    universe_map(ui),
                ).into()
            }

            CanonicalVarKind::PlaceholderConst(
                ty::PlaceholderConst { universe, name },
            ) => {
                let universe_mapped = universe_map(universe);
                let placeholder_mapped = ty::PlaceholderConst {
                    universe: universe_mapped,
                    name,
                };
                self.tcx.mk_const(
                    ty::Const {
                        val: ConstValue::Placeholder(placeholder_mapped),
                        ty: self.tcx.types.err, // FIXME(const_generics)
                    }
                ).into()
            }
        }
    }
}

CloneTypeFoldableAndLiftImpls! {
    crate::infer::canonical::Certainty,
    crate::infer::canonical::CanonicalVarInfo,
    crate::infer::canonical::CanonicalVarKind,
}

CloneTypeFoldableImpls! {
    for <'tcx> {
        crate::infer::canonical::CanonicalVarInfos<'tcx>,
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx, C> TypeFoldable<'tcx> for Canonical<'tcx, C> {
        max_universe,
        variables,
        value,
    } where C: TypeFoldable<'tcx>
}

BraceStructLiftImpl! {
    impl<'a, 'tcx, T> Lift<'tcx> for Canonical<'a, T> {
        type Lifted = Canonical<'tcx, T::Lifted>;
        max_universe, variables, value
    } where T: Lift<'tcx>
}

impl<'tcx> CanonicalVarValues<'tcx> {
    pub fn len(&self) -> usize {
        self.var_values.len()
    }

    /// Makes an identity substitution from this one: each bound var
    /// is matched to the same bound var, preserving the original kinds.
    /// For example, if we have:
    /// `self.var_values == [Type(u32), Lifetime('a), Type(u64)]`
    /// we'll return a substitution `subst` with:
    /// `subst.var_values == [Type(^0), Lifetime(^1), Type(^2)]`.
    pub fn make_identity(&self, tcx: TyCtxt<'tcx>) -> Self {
        use crate::ty::subst::UnpackedKind;

        CanonicalVarValues {
            var_values: self.var_values.iter()
                .zip(0..)
                .map(|(kind, i)| match kind.unpack() {
                    UnpackedKind::Type(..) => tcx.mk_ty(
                        ty::Bound(ty::INNERMOST, ty::BoundVar::from_u32(i).into())
                    ).into(),
                    UnpackedKind::Lifetime(..) => tcx.mk_region(
                        ty::ReLateBound(ty::INNERMOST, ty::BoundRegion::BrAnon(i))
                    ).into(),
                    UnpackedKind::Const(ct) => {
                        tcx.mk_const(ty::Const {
                            ty: ct.ty,
                            val: ConstValue::Infer(
                                InferConst::Canonical(ty::INNERMOST, ty::BoundVar::from_u32(i))
                            ),
                        }).into()
                    }
                })
                .collect()
        }
    }
}

impl<'a, 'tcx> IntoIterator for &'a CanonicalVarValues<'tcx> {
    type Item = Kind<'tcx>;
    type IntoIter = ::std::iter::Cloned<::std::slice::Iter<'a, Kind<'tcx>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.var_values.iter().cloned()
    }
}

BraceStructLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for CanonicalVarValues<'a> {
        type Lifted = CanonicalVarValues<'tcx>;
        var_values,
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for CanonicalVarValues<'tcx> {
        var_values,
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx, R> TypeFoldable<'tcx> for QueryResponse<'tcx, R> {
        var_values, region_constraints, certainty, value
    } where R: TypeFoldable<'tcx>,
}

BraceStructLiftImpl! {
    impl<'a, 'tcx, R> Lift<'tcx> for QueryResponse<'a, R> {
        type Lifted = QueryResponse<'tcx, R::Lifted>;
        var_values, region_constraints, certainty, value
    } where R: Lift<'tcx>
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for QueryRegionConstraints<'tcx> {
        outlives, member_constraints
    }
}

BraceStructLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for QueryRegionConstraints<'a> {
        type Lifted = QueryRegionConstraints<'tcx>;
        outlives, member_constraints
    }
}

impl<'tcx> Index<BoundVar> for CanonicalVarValues<'tcx> {
    type Output = Kind<'tcx>;

    fn index(&self, value: BoundVar) -> &Kind<'tcx> {
        &self.var_values[value]
    }
}
