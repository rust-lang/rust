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
//! [c]: https://rust-lang-nursery.github.io/rustc-guide/traits/canonicalization.html

use infer::{InferCtxt, RegionVariableOrigin, TypeVariableOrigin};
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_data_structures::small_vec::SmallVec;
use rustc_data_structures::sync::Lrc;
use serialize::UseSpecializedDecodable;
use std::ops::Index;
use syntax::source_map::Span;
use ty::fold::TypeFoldable;
use ty::subst::Kind;
use ty::{self, CanonicalVar, Lift, Region, Slice, TyCtxt};

mod canonicalizer;

pub mod query_result;

mod substitute;

/// A "canonicalized" type `V` is one where all free inference
/// variables have been rewriten to "canonical vars". These are
/// numbered starting from 0 in order of first appearance.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, RustcDecodable, RustcEncodable)]
pub struct Canonical<'gcx, V> {
    pub variables: CanonicalVarInfos<'gcx>,
    pub value: V,
}

pub type CanonicalVarInfos<'gcx> = &'gcx Slice<CanonicalVarInfo>;

impl<'gcx> UseSpecializedDecodable for CanonicalVarInfos<'gcx> {}

/// A set of values corresponding to the canonical variables from some
/// `Canonical`. You can give these values to
/// `canonical_value.substitute` to substitute them into the canonical
/// value at the right places.
///
/// When you canonicalize a value `V`, you get back one of these
/// vectors with the original values that were replaced by canonical
/// variables. You will need to supply it later to instantiate the
/// canonicalized query response.
#[derive(Clone, Debug, PartialEq, Eq, Hash, RustcDecodable, RustcEncodable)]
pub struct CanonicalVarValues<'tcx> {
    pub var_values: IndexVec<CanonicalVar, Kind<'tcx>>,
}

/// Like CanonicalVarValues, but for use in places where a SmallVec is
/// appropriate.
pub type SmallCanonicalVarValues<'tcx> = SmallVec<[Kind<'tcx>; 8]>;

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
    pub region_constraints: Vec<QueryRegionConstraint<'tcx>>,
    pub certainty: Certainty,
    pub value: R,
}

pub type Canonicalized<'gcx, V> = Canonical<'gcx, <V as Lift<'gcx>>::Lifted>;

pub type CanonicalizedQueryResult<'gcx, T> =
    Lrc<Canonical<'gcx, QueryResult<'gcx, <T as Lift<'gcx>>::Lifted>>>;

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

pub type QueryRegionConstraint<'tcx> = ty::Binder<ty::OutlivesPredicate<Kind<'tcx>, Region<'tcx>>>;

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
    fn fresh_inference_vars_for_canonical_vars(
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
    fn fresh_inference_var_for_canonical_var(
        &self,
        span: Span,
        cv_info: CanonicalVarInfo,
    ) -> Kind<'tcx> {
        match cv_info.kind {
            CanonicalVarKind::Ty(ty_kind) => {
                let ty = match ty_kind {
                    CanonicalTyVarKind::General => {
                        self.next_ty_var(TypeVariableOrigin::MiscVariable(span))
                    }

                    CanonicalTyVarKind::Int => self.tcx.mk_int_var(self.next_int_var_id()),

                    CanonicalTyVarKind::Float => self.tcx.mk_float_var(self.next_float_var_id()),
                };
                ty.into()
            }

            CanonicalVarKind::Region => self
                .next_region_var(RegionVariableOrigin::MiscVariable(span))
                .into(),
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
