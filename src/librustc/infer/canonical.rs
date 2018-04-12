// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
//! We can then do queries using T2. These will give back constriants
//! on the canonical variables which can be translated, using the map
//! M, into constraints in our source context. This process of
//! translating the results back is done by the
//! `instantiate_query_result` method.
//!
//! For a more detailed look at what is happening here, check
//! out the [chapter in the rustc guide][c].
//!
//! [c]: https://rust-lang-nursery.github.io/rustc-guide/traits-canonicalization.html

use infer::{InferCtxt, InferOk, InferResult, RegionVariableOrigin, TypeVariableOrigin};
use rustc_data_structures::indexed_vec::Idx;
use serialize::UseSpecializedDecodable;
use std::fmt::Debug;
use std::ops::Index;
use std::sync::atomic::Ordering;
use syntax::codemap::Span;
use traits::{Obligation, ObligationCause, PredicateObligation};
use ty::{self, CanonicalVar, Lift, Region, Slice, Ty, TyCtxt, TypeFlags};
use ty::subst::{Kind, UnpackedKind};
use ty::fold::{TypeFoldable, TypeFolder};
use util::captures::Captures;

use rustc_data_structures::indexed_vec::IndexVec;
use rustc_data_structures::fx::FxHashMap;

/// A "canonicalized" type `V` is one where all free inference
/// variables have been rewriten to "canonical vars". These are
/// numbered starting from 0 in order of first appearance.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, RustcDecodable, RustcEncodable)]
pub struct Canonical<'gcx, V> {
    pub variables: CanonicalVarInfos<'gcx>,
    pub value: V,
}

pub type CanonicalVarInfos<'gcx> = &'gcx Slice<CanonicalVarInfo>;

impl<'gcx> UseSpecializedDecodable for CanonicalVarInfos<'gcx> { }

/// A set of values corresponding to the canonical variables from some
/// `Canonical`. You can give these values to
/// `canonical_value.substitute` to substitute them into the canonical
/// value at the right places.
///
/// When you canonicalize a value `V`, you get back one of these
/// vectors with the original values that were replaced by canonical
/// variables.
///
/// You can also use `infcx.fresh_inference_vars_for_canonical_vars`
/// to get back a `CanonicalVarValues` containing fresh inference
/// variables.
#[derive(Clone, Debug, PartialEq, Eq, Hash, RustcDecodable, RustcEncodable)]
pub struct CanonicalVarValues<'tcx> {
    pub var_values: IndexVec<CanonicalVar, Kind<'tcx>>,
}

/// Information about a canonical variable that is included with the
/// canonical value. This is sufficient information for code to create
/// a copy of the canonical value in some other inference context,
/// with fresh inference variables replacing the canonical values.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, RustcDecodable, RustcEncodable)]
pub struct CanonicalVarInfo {
    pub kind: CanonicalVarKind,
}

/// Describes the "kind" of the canonical variable. This is a "kind"
/// in the type-theory sense of the term -- i.e., a "meta" type system
/// that analyzes type-like values.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, RustcDecodable, RustcEncodable)]
pub enum CanonicalVarKind {
    /// Some kind of type inference variable.
    Ty(CanonicalTyVarKind),

    /// Region variable `'?R`.
    Region,
}

/// Rust actually has more than one category of type variables;
/// notably, the type variables we create for literals (e.g., 22 or
/// 22.) can only be instantiated with integral/float types (e.g.,
/// usize or f32). In order to faithfully reproduce a type, we need to
/// know what set of types a given type variable can be unified with.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, RustcDecodable, RustcEncodable)]
pub enum CanonicalTyVarKind {
    /// General type variable `?T` that can be unified with arbitrary types.
    General,

    /// Integral type variable `?I` (that can only be unified with integral types).
    Int,

    /// Floating-point type variable `?F` (that can only be unified with float types).
    Float,
}

/// After we execute a query with a canonicalized key, we get back a
/// `Canonical<QueryResult<..>>`. You can use
/// `instantiate_query_result` to access the data in this result.
#[derive(Clone, Debug)]
pub struct QueryResult<'tcx, R> {
    pub var_values: CanonicalVarValues<'tcx>,
    pub region_constraints: QueryRegionConstraints<'tcx>,
    pub certainty: Certainty,
    pub value: R,
}

/// Indicates whether or not we were able to prove the query to be
/// true.
#[derive(Copy, Clone, Debug)]
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

impl<'tcx, R> QueryResult<'tcx, R> {
    pub fn is_proven(&self) -> bool {
        self.certainty.is_proven()
    }

    pub fn is_ambiguous(&self) -> bool {
        !self.is_proven()
    }
}

impl<'tcx, R> Canonical<'tcx, QueryResult<'tcx, R>> {
    pub fn is_proven(&self) -> bool {
        self.value.is_proven()
    }

    pub fn is_ambiguous(&self) -> bool {
        !self.is_proven()
    }
}

/// Subset of `RegionConstraintData` produced by trait query.
#[derive(Clone, Debug, Default)]
pub struct QueryRegionConstraints<'tcx> {
    pub region_outlives: Vec<(Region<'tcx>, Region<'tcx>)>,
    pub ty_outlives: Vec<(Ty<'tcx>, Region<'tcx>)>,
}

/// Trait implemented by values that can be canonicalized. It mainly
/// serves to identify the interning table we will use.
pub trait Canonicalize<'gcx: 'tcx, 'tcx>: TypeFoldable<'tcx> + Lift<'gcx> {
    type Canonicalized: 'gcx + Debug;

    /// After a value has been fully canonicalized and lifted, this
    /// method will allocate it in a global arena.
    fn intern(
        gcx: TyCtxt<'_, 'gcx, 'gcx>,
        value: Canonical<'gcx, Self::Lifted>,
    ) -> Self::Canonicalized;
}

impl<'cx, 'gcx, 'tcx> InferCtxt<'cx, 'gcx, 'tcx> {
    /// Creates a substitution S for the canonical value with fresh
    /// inference variables and applies it to the canonical value.
    /// Returns both the instantiated result *and* the substitution S.
    ///
    /// This is useful at the start of a query: it basically brings
    /// the canonical value "into scope" within your new infcx. At the
    /// end of processing, the substitution S (once canonicalized)
    /// then represents the values that you computed for each of the
    /// canonical inputs to your query.
    pub fn instantiate_canonical_with_fresh_inference_vars<T>(
        &self,
        span: Span,
        canonical: &Canonical<'tcx, T>,
    ) -> (T, CanonicalVarValues<'tcx>)
    where
        T: TypeFoldable<'tcx>,
    {
        let canonical_inference_vars =
            self.fresh_inference_vars_for_canonical_vars(span, canonical.variables);
        let result = canonical.substitute(self.tcx, &canonical_inference_vars);
        (result, canonical_inference_vars)
    }

    /// Given the "infos" about the canonical variables from some
    /// canonical, creates fresh inference variables with the same
    /// characteristics. You can then use `substitute` to instantiate
    /// the canonical variable with these inference variables.
    pub fn fresh_inference_vars_for_canonical_vars(
        &self,
        span: Span,
        variables: &Slice<CanonicalVarInfo>,
    ) -> CanonicalVarValues<'tcx> {
        let var_values: IndexVec<CanonicalVar, Kind<'tcx>> = variables
            .iter()
            .map(|info| self.fresh_inference_var_for_canonical_var(span, *info))
            .collect();

        CanonicalVarValues { var_values }
    }

    /// Given the "info" about a canonical variable, creates a fresh
    /// inference variable with the same characteristics.
    pub fn fresh_inference_var_for_canonical_var(
        &self,
        span: Span,
        cv_info: CanonicalVarInfo,
    ) -> Kind<'tcx> {
        match cv_info.kind {
            CanonicalVarKind::Ty(ty_kind) => {
                let ty = match ty_kind {
                    CanonicalTyVarKind::General => {
                        self.next_ty_var(
                            TypeVariableOrigin::MiscVariable(span),
                        )
                    }

                    CanonicalTyVarKind::Int => self.tcx.mk_int_var(self.next_int_var_id()),

                    CanonicalTyVarKind::Float => self.tcx.mk_float_var(self.next_float_var_id()),
                };
                Kind::from(ty)
            }

            CanonicalVarKind::Region => {
                Kind::from(self.next_region_var(RegionVariableOrigin::MiscVariable(span)))
            }
        }
    }

    /// Given the (canonicalized) result to a canonical query,
    /// instantiates the result so it can be used, plugging in the
    /// values from the canonical query. (Note that the result may
    /// have been ambiguous; you should check the certainty level of
    /// the query before applying this function.)
    ///
    /// To get a good understanding of what is happening here, check
    /// out the [chapter in the rustc guide][c].
    ///
    /// [c]: https://rust-lang-nursery.github.io/rustc-guide/traits-canonicalization.html#processing-the-canonicalized-query-result
    pub fn instantiate_query_result<R>(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        original_values: &CanonicalVarValues<'tcx>,
        query_result: &Canonical<'tcx, QueryResult<'tcx, R>>,
    ) -> InferResult<'tcx, R>
    where
        R: Debug + TypeFoldable<'tcx>,
    {
        debug!(
            "instantiate_query_result(original_values={:#?}, query_result={:#?})",
            original_values, query_result,
        );

        // Every canonical query result includes values for each of
        // the inputs to the query. Therefore, we begin by unifying
        // these values with the original inputs that were
        // canonicalized.
        let result_values = &query_result.value.var_values;
        assert_eq!(original_values.len(), result_values.len());

        // Quickly try to find initial values for the canonical
        // variables in the result in terms of the query. We do this
        // by iterating down the values that the query gave to each of
        // the canonical inputs. If we find that one of those values
        // is directly equal to one of the canonical variables in the
        // result, then we can type the corresponding value from the
        // input. See the example above.
        let mut opt_values: IndexVec<CanonicalVar, Option<Kind<'tcx>>> =
            IndexVec::from_elem_n(None, query_result.variables.len());

        // In terms of our example above, we are iterating over pairs like:
        // [(?A, Vec<?0>), ('static, '?1), (?B, ?0)]
        for (original_value, result_value) in original_values.iter().zip(result_values) {
            match result_value.unpack() {
                UnpackedKind::Type(result_value) => {
                    // e.g., here `result_value` might be `?0` in the example above...
                    if let ty::TyInfer(ty::InferTy::CanonicalTy(index)) = result_value.sty {
                        // in which case we would set `canonical_vars[0]` to `Some(?U)`.
                        opt_values[index] = Some(original_value);
                    }
                }
                UnpackedKind::Lifetime(result_value) => {
                    // e.g., here `result_value` might be `'?1` in the example above...
                    if let &ty::RegionKind::ReCanonical(index) = result_value {
                        // in which case we would set `canonical_vars[0]` to `Some('static)`.
                        opt_values[index] = Some(original_value);
                    }
                }
            }
        }

        // Create a result substitution: if we found a value for a
        // given variable in the loop above, use that. Otherwise, use
        // a fresh inference variable.
        let result_subst = &CanonicalVarValues {
            var_values: query_result
                .variables
                .iter()
                .enumerate()
                .map(|(index, info)| match opt_values[CanonicalVar::new(index)] {
                    Some(k) => k,
                    None => self.fresh_inference_var_for_canonical_var(cause.span, *info),
                })
                .collect(),
        };

        // Unify the original values for the canonical variables in
        // the input with the value found in the query
        // post-substitution. Often, but not always, this is a no-op,
        // because we already found the mapping in the first step.
        let substituted_values = |index: CanonicalVar| -> Kind<'tcx> {
            query_result.substitute_projected(self.tcx, result_subst, |v| &v.var_values[index])
        };
        let mut obligations =
            self.unify_canonical_vars(cause, param_env, original_values, substituted_values)?
                .into_obligations();

        obligations.extend(self.query_region_constraints_into_obligations(
            cause,
            param_env,
            &query_result.value.region_constraints,
            result_subst,
        ));

        let user_result: R =
            query_result.substitute_projected(self.tcx, result_subst, |q_r| &q_r.value);

        Ok(InferOk {
            value: user_result,
            obligations,
        })
    }

    /// Converts the region constraints resulting from a query into an
    /// iterator of obligations.
    fn query_region_constraints_into_obligations<'a>(
        &'a self,
        cause: &'a ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        unsubstituted_region_constraints: &'a QueryRegionConstraints<'tcx>,
        result_subst: &'a CanonicalVarValues<'tcx>,
    ) -> impl Iterator<Item = PredicateObligation<'tcx>> + Captures<'gcx> + 'a {
        let QueryRegionConstraints {
            region_outlives,
            ty_outlives,
        } = unsubstituted_region_constraints;

        let region_obligations = region_outlives.iter().map(move |(r1, r2)| {
            let r1 = substitute_value(self.tcx, result_subst, r1);
            let r2 = substitute_value(self.tcx, result_subst, r2);
            Obligation::new(
                cause.clone(),
                param_env,
                ty::Predicate::RegionOutlives(ty::Binder(ty::OutlivesPredicate(r1, r2))),
            )
        });

        let ty_obligations = ty_outlives.iter().map(move |(t1, r2)| {
            let t1 = substitute_value(self.tcx, result_subst, t1);
            let r2 = substitute_value(self.tcx, result_subst, r2);
            Obligation::new(
                cause.clone(),
                param_env,
                ty::Predicate::TypeOutlives(ty::Binder(ty::OutlivesPredicate(t1, r2))),
            )
        });

        region_obligations.chain(ty_obligations)
    }

    /// Given two sets of values for the same set of canonical variables, unify them.
    /// The second set is produced lazilly by supplying indices from the first set.
    fn unify_canonical_vars(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        variables1: &CanonicalVarValues<'tcx>,
        variables2: impl Fn(CanonicalVar) -> Kind<'tcx>,
    ) -> InferResult<'tcx, ()> {
        self.commit_if_ok(|_| {
            let mut obligations = vec![];
            for (index, value1) in variables1.var_values.iter_enumerated() {
                let value2 = variables2(index);

                match (value1.unpack(), value2.unpack()) {
                    (UnpackedKind::Type(v1), UnpackedKind::Type(v2)) => {
                        obligations
                            .extend(self.at(cause, param_env).eq(v1, v2)?.into_obligations());
                    }
                    (
                        UnpackedKind::Lifetime(ty::ReErased),
                        UnpackedKind::Lifetime(ty::ReErased),
                    ) => {
                        // no action needed
                    }
                    (UnpackedKind::Lifetime(v1), UnpackedKind::Lifetime(v2)) => {
                        obligations
                            .extend(self.at(cause, param_env).eq(v1, v2)?.into_obligations());
                    }
                    _ => {
                        bug!("kind mismatch, cannot unify {:?} and {:?}", value1, value2,);
                    }
                }
            }
            Ok(InferOk {
                value: (),
                obligations,
            })
        })
    }

    /// Canonicalizes a query value `V`. When we canonicalize a query,
    /// we not only canonicalize unbound inference variables, but we
    /// *also* replace all free regions whatsoever. So for example a
    /// query like `T: Trait<'static>` would be canonicalized to
    ///
    /// ```text
    /// T: Trait<'?0>
    /// ```
    ///
    /// with a mapping M that maps `'?0` to `'static`.
    ///
    /// To get a good understanding of what is happening here, check
    /// out the [chapter in the rustc guide][c].
    ///
    /// [c]: https://rust-lang-nursery.github.io/rustc-guide/traits-canonicalization.html#canonicalizing-the-query
    pub fn canonicalize_query<V>(&self, value: &V) -> (V::Canonicalized, CanonicalVarValues<'tcx>)
    where
        V: Canonicalize<'gcx, 'tcx>,
    {
        self.tcx.sess.perf_stats.queries_canonicalized.fetch_add(1, Ordering::Relaxed);

        Canonicalizer::canonicalize(
            value,
            Some(self),
            self.tcx,
            CanonicalizeAllFreeRegions(true),
        )
    }

    /// Canonicalizes a query *response* `V`. When we canonicalize a
    /// query response, we only canonicalize unbound inference
    /// variables, and we leave other free regions alone. So,
    /// continuing with the example from `canonicalize_query`, if
    /// there was an input query `T: Trait<'static>`, it would have
    /// been canonicalized to
    ///
    /// ```text
    /// T: Trait<'?0>
    /// ```
    ///
    /// with a mapping M that maps `'?0` to `'static`. But if we found that there
    /// exists only one possible impl of `Trait`, and it looks like
    ///
    ///     impl<T> Trait<'static> for T { .. }
    ///
    /// then we would prepare a query result R that (among other
    /// things) includes a mapping to `'?0 := 'static`. When
    /// canonicalizing this query result R, we would leave this
    /// reference to `'static` alone.
    ///
    /// To get a good understanding of what is happening here, check
    /// out the [chapter in the rustc guide][c].
    ///
    /// [c]: https://rust-lang-nursery.github.io/rustc-guide/traits-canonicalization.html#canonicalizing-the-query-result
    pub fn canonicalize_response<V>(
        &self,
        value: &V,
    ) -> (V::Canonicalized, CanonicalVarValues<'tcx>)
    where
        V: Canonicalize<'gcx, 'tcx>,
    {
        Canonicalizer::canonicalize(
            value,
            Some(self),
            self.tcx,
            CanonicalizeAllFreeRegions(false),
        )
    }
}

/// If this flag is true, then all free regions will be replaced with
/// a canonical var. This is used to make queries as generic as
/// possible. For example, the query `F: Foo<'static>` would be
/// canonicalized to `F: Foo<'0>`.
struct CanonicalizeAllFreeRegions(bool);

struct Canonicalizer<'cx, 'gcx: 'tcx, 'tcx: 'cx> {
    infcx: Option<&'cx InferCtxt<'cx, 'gcx, 'tcx>>,
    tcx: TyCtxt<'cx, 'gcx, 'tcx>,
    variables: IndexVec<CanonicalVar, CanonicalVarInfo>,
    indices: FxHashMap<Kind<'tcx>, CanonicalVar>,
    var_values: IndexVec<CanonicalVar, Kind<'tcx>>,
    canonicalize_all_free_regions: CanonicalizeAllFreeRegions,
    needs_canonical_flags: TypeFlags,
}

impl<'cx, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for Canonicalizer<'cx, 'gcx, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> {
        self.tcx
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match *r {
            ty::ReLateBound(..) => {
                // leave bound regions alone
                r
            }

            ty::ReVar(vid) => {
                let r = self.infcx
                    .unwrap()
                    .borrow_region_constraints()
                    .opportunistic_resolve_var(self.tcx, vid);
                let info = CanonicalVarInfo {
                    kind: CanonicalVarKind::Region,
                };
                debug!(
                    "canonical: region var found with vid {:?}, \
                     opportunistically resolved to {:?}",
                    vid, r
                );
                let cvar = self.canonical_var(info, Kind::from(r));
                self.tcx().mk_region(ty::ReCanonical(cvar))
            }

            ty::ReStatic
            | ty::ReEarlyBound(..)
            | ty::ReFree(_)
            | ty::ReScope(_)
            | ty::ReSkolemized(..)
            | ty::ReEmpty
            | ty::ReErased => {
                if self.canonicalize_all_free_regions.0 {
                    let info = CanonicalVarInfo {
                        kind: CanonicalVarKind::Region,
                    };
                    let cvar = self.canonical_var(info, Kind::from(r));
                    self.tcx().mk_region(ty::ReCanonical(cvar))
                } else {
                    r
                }
            }

            ty::ReClosureBound(..) | ty::ReCanonical(_) => {
                bug!("canonical region encountered during canonicalization")
            }
        }
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        match t.sty {
            ty::TyInfer(ty::TyVar(_)) => self.canonicalize_ty_var(CanonicalTyVarKind::General, t),

            ty::TyInfer(ty::IntVar(_)) => self.canonicalize_ty_var(CanonicalTyVarKind::Int, t),

            ty::TyInfer(ty::FloatVar(_)) => self.canonicalize_ty_var(CanonicalTyVarKind::Float, t),

            ty::TyInfer(ty::FreshTy(_))
            | ty::TyInfer(ty::FreshIntTy(_))
            | ty::TyInfer(ty::FreshFloatTy(_)) => {
                bug!("encountered a fresh type during canonicalization")
            }

            ty::TyInfer(ty::CanonicalTy(_)) => {
                bug!("encountered a canonical type during canonicalization")
            }

            ty::TyClosure(..)
            | ty::TyGenerator(..)
            | ty::TyGeneratorWitness(..)
            | ty::TyBool
            | ty::TyChar
            | ty::TyInt(..)
            | ty::TyUint(..)
            | ty::TyFloat(..)
            | ty::TyAdt(..)
            | ty::TyStr
            | ty::TyError
            | ty::TyArray(..)
            | ty::TySlice(..)
            | ty::TyRawPtr(..)
            | ty::TyRef(..)
            | ty::TyFnDef(..)
            | ty::TyFnPtr(_)
            | ty::TyDynamic(..)
            | ty::TyNever
            | ty::TyTuple(..)
            | ty::TyProjection(..)
            | ty::TyForeign(..)
            | ty::TyParam(..)
            | ty::TyAnon(..) => {
                if t.flags.intersects(self.needs_canonical_flags) {
                    t.super_fold_with(self)
                } else {
                    t
                }
            }
        }
    }
}

impl<'cx, 'gcx, 'tcx> Canonicalizer<'cx, 'gcx, 'tcx> {
    /// The main `canonicalize` method, shared impl of
    /// `canonicalize_query` and `canonicalize_response`.
    fn canonicalize<V>(
        value: &V,
        infcx: Option<&'cx InferCtxt<'cx, 'gcx, 'tcx>>,
        tcx: TyCtxt<'cx, 'gcx, 'tcx>,
        canonicalize_all_free_regions: CanonicalizeAllFreeRegions,
    ) -> (V::Canonicalized, CanonicalVarValues<'tcx>)
    where
        V: Canonicalize<'gcx, 'tcx>,
    {
        debug_assert!(
            !value.has_type_flags(TypeFlags::HAS_CANONICAL_VARS),
            "canonicalizing a canonical value: {:?}",
            value,
        );

        let needs_canonical_flags = if canonicalize_all_free_regions.0 {
            TypeFlags::HAS_FREE_REGIONS | TypeFlags::KEEP_IN_LOCAL_TCX
        } else {
            TypeFlags::KEEP_IN_LOCAL_TCX
        };

        let gcx = tcx.global_tcx();

        // Fast path: nothing that needs to be canonicalized.
        if !value.has_type_flags(needs_canonical_flags) {
            let out_value = gcx.lift(value).unwrap();
            let canon_value = V::intern(
                gcx,
                Canonical {
                    variables: Slice::empty(),
                    value: out_value,
                },
            );
            let values = CanonicalVarValues {
                var_values: IndexVec::default(),
            };
            return (canon_value, values);
        }

        let mut canonicalizer = Canonicalizer {
            infcx,
            tcx,
            canonicalize_all_free_regions,
            needs_canonical_flags,
            variables: IndexVec::default(),
            indices: FxHashMap::default(),
            var_values: IndexVec::default(),
        };
        let out_value = value.fold_with(&mut canonicalizer);

        // Once we have canonicalized `out_value`, it should not
        // contain anything that ties it to this inference context
        // anymore, so it should live in the global arena.
        let out_value = gcx.lift(&out_value).unwrap_or_else(|| {
            bug!(
                "failed to lift `{:?}`, canonicalized from `{:?}`",
                out_value,
                value
            )
        });

        let canonical_variables = tcx.intern_canonical_var_infos(&canonicalizer.variables.raw);

        let canonical_value = V::intern(
            gcx,
            Canonical {
                variables: canonical_variables,
                value: out_value,
            },
        );
        let canonical_var_values = CanonicalVarValues {
            var_values: canonicalizer.var_values,
        };
        (canonical_value, canonical_var_values)
    }

    /// Creates a canonical variable replacing `kind` from the input,
    /// or returns an existing variable if `kind` has already been
    /// seen. `kind` is expected to be an unbound variable (or
    /// potentially a free region).
    fn canonical_var(&mut self, info: CanonicalVarInfo, kind: Kind<'tcx>) -> CanonicalVar {
        let Canonicalizer {
            indices,
            variables,
            var_values,
            ..
        } = self;

        indices
            .entry(kind)
            .or_insert_with(|| {
                let cvar1 = variables.push(info);
                let cvar2 = var_values.push(kind);
                assert_eq!(cvar1, cvar2);
                cvar1
            })
            .clone()
    }

    /// Given a type variable `ty_var` of the given kind, first check
    /// if `ty_var` is bound to anything; if so, canonicalize
    /// *that*. Otherwise, create a new canonical variable for
    /// `ty_var`.
    fn canonicalize_ty_var(&mut self, ty_kind: CanonicalTyVarKind, ty_var: Ty<'tcx>) -> Ty<'tcx> {
        let infcx = self.infcx.expect("encountered ty-var without infcx");
        let bound_to = infcx.shallow_resolve(ty_var);
        if bound_to != ty_var {
            self.fold_ty(bound_to)
        } else {
            let info = CanonicalVarInfo {
                kind: CanonicalVarKind::Ty(ty_kind),
            };
            let cvar = self.canonical_var(info, Kind::from(ty_var));
            self.tcx().mk_infer(ty::InferTy::CanonicalTy(cvar))
        }
    }
}

impl<'tcx, V> Canonical<'tcx, V> {
    /// Instantiate the wrapped value, replacing each canonical value
    /// with the value given in `var_values`.
    fn substitute(&self, tcx: TyCtxt<'_, '_, 'tcx>, var_values: &CanonicalVarValues<'tcx>) -> V
    where
        V: TypeFoldable<'tcx>,
    {
        self.substitute_projected(tcx, var_values, |value| value)
    }

    /// Invoke `projection_fn` with `self.value` to get a value V that
    /// is expressed in terms of the same canonical variables bound in
    /// `self`. Apply the substitution `var_values` to this value V,
    /// replacing each of the canonical variables.
    fn substitute_projected<T>(
        &self,
        tcx: TyCtxt<'_, '_, 'tcx>,
        var_values: &CanonicalVarValues<'tcx>,
        projection_fn: impl FnOnce(&V) -> &T,
    ) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        assert_eq!(self.variables.len(), var_values.var_values.len());
        let value = projection_fn(&self.value);
        substitute_value(tcx, var_values, value)
    }
}

/// Substitute the values from `var_values` into `value`. `var_values`
/// must be values for the set of cnaonical variables that appear in
/// `value`.
fn substitute_value<'a, 'tcx, T>(
    tcx: TyCtxt<'_, '_, 'tcx>,
    var_values: &CanonicalVarValues<'tcx>,
    value: &'a T,
) -> T
where
    T: TypeFoldable<'tcx>,
{
    if var_values.var_values.is_empty() {
        debug_assert!(!value.has_type_flags(TypeFlags::HAS_CANONICAL_VARS));
        value.clone()
    } else if !value.has_type_flags(TypeFlags::HAS_CANONICAL_VARS) {
        value.clone()
    } else {
        value.fold_with(&mut CanonicalVarValuesSubst { tcx, var_values })
    }
}

struct CanonicalVarValuesSubst<'cx, 'gcx: 'tcx, 'tcx: 'cx> {
    tcx: TyCtxt<'cx, 'gcx, 'tcx>,
    var_values: &'cx CanonicalVarValues<'tcx>,
}

impl<'cx, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for CanonicalVarValuesSubst<'cx, 'gcx, 'tcx> {
    fn tcx(&self) -> TyCtxt<'_, 'gcx, 'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        match t.sty {
            ty::TyInfer(ty::InferTy::CanonicalTy(c)) => {
                match self.var_values.var_values[c].unpack() {
                    UnpackedKind::Type(ty) => ty,
                    r => bug!("{:?} is a type but value is {:?}", c, r),
                }
            }
            _ => {
                if !t.has_type_flags(TypeFlags::HAS_CANONICAL_VARS) {
                    t
                } else {
                    t.super_fold_with(self)
                }
            }
        }
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match r {
            ty::RegionKind::ReCanonical(c) => match self.var_values.var_values[*c].unpack() {
                UnpackedKind::Lifetime(l) => l,
                r => bug!("{:?} is a region but value is {:?}", c, r),
            },
            _ => r.super_fold_with(self),
        }
    }
}

CloneTypeFoldableAndLiftImpls! {
    ::infer::canonical::Certainty,
    ::infer::canonical::CanonicalVarInfo,
    ::infer::canonical::CanonicalVarKind,
}

CloneTypeFoldableImpls! {
    for <'tcx> {
        ::infer::canonical::CanonicalVarInfos<'tcx>,
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx, C> TypeFoldable<'tcx> for Canonical<'tcx, C> {
        variables,
        value,
    } where C: TypeFoldable<'tcx>
}

BraceStructLiftImpl! {
    impl<'a, 'tcx, T> Lift<'tcx> for Canonical<'a, T> {
        type Lifted = Canonical<'tcx, T::Lifted>;
        variables, value
    } where T: Lift<'tcx>
}

impl<'tcx> CanonicalVarValues<'tcx> {
    fn iter<'a>(&'a self) -> impl Iterator<Item = Kind<'tcx>> + 'a {
        self.var_values.iter().cloned()
    }

    fn len(&self) -> usize {
        self.var_values.len()
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
    impl<'tcx> TypeFoldable<'tcx> for QueryRegionConstraints<'tcx> {
        region_outlives, ty_outlives
    }
}

BraceStructLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for QueryRegionConstraints<'a> {
        type Lifted = QueryRegionConstraints<'tcx>;
        region_outlives, ty_outlives
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx, R> TypeFoldable<'tcx> for QueryResult<'tcx, R> {
        var_values, region_constraints, certainty, value
    } where R: TypeFoldable<'tcx>,
}

BraceStructLiftImpl! {
    impl<'a, 'tcx, R> Lift<'tcx> for QueryResult<'a, R> {
        type Lifted = QueryResult<'tcx, R::Lifted>;
        var_values, region_constraints, certainty, value
    } where R: Lift<'tcx>
}

impl<'tcx> Index<CanonicalVar> for CanonicalVarValues<'tcx> {
    type Output = Kind<'tcx>;

    fn index(&self, value: CanonicalVar) -> &Kind<'tcx> {
        &self.var_values[value]
    }
}
