// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use borrow_check::nll::constraints::OutlivesConstraint;
use borrow_check::nll::type_check::{BorrowCheckContext, Locations};
use borrow_check::nll::universal_regions::UniversalRegions;
use borrow_check::nll::ToRegionVid;
use rustc::infer::canonical::{Canonical, CanonicalVarInfos};
use rustc::infer::{InferCtxt, NLLRegionVariableOrigin};
use rustc::traits::query::Fallible;
use rustc::ty::fold::{TypeFoldable, TypeVisitor};
use rustc::ty::relate::{self, Relate, RelateResult, TypeRelation};
use rustc::ty::subst::Kind;
use rustc::ty::{self, CanonicalTy, CanonicalVar, RegionVid, Ty, TyCtxt};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexed_vec::IndexVec;

/// Adds sufficient constraints to ensure that `a <: b`.
pub(super) fn sub_types<'tcx>(
    infcx: &InferCtxt<'_, '_, 'tcx>,
    a: Ty<'tcx>,
    b: Ty<'tcx>,
    locations: Locations,
    borrowck_context: Option<&mut BorrowCheckContext<'_, 'tcx>>,
) -> Fallible<()> {
    debug!("sub_types(a={:?}, b={:?}, locations={:?})", a, b, locations);
    TypeRelating::new(
        infcx,
        ty::Variance::Covariant,
        locations,
        borrowck_context,
        ty::List::empty(),
    ).relate(&a, &b)?;
    Ok(())
}

/// Adds sufficient constraints to ensure that `a == b`.
pub(super) fn eq_types<'tcx>(
    infcx: &InferCtxt<'_, '_, 'tcx>,
    a: Ty<'tcx>,
    b: Ty<'tcx>,
    locations: Locations,
    borrowck_context: Option<&mut BorrowCheckContext<'_, 'tcx>>,
) -> Fallible<()> {
    debug!("eq_types(a={:?}, b={:?}, locations={:?})", a, b, locations);
    TypeRelating::new(
        infcx,
        ty::Variance::Invariant,
        locations,
        borrowck_context,
        ty::List::empty(),
    ).relate(&a, &b)?;
    Ok(())
}

/// Adds sufficient constraints to ensure that `a <: b`, where `b` is
/// a user-given type (which means it may have canonical variables
/// encoding things like `_`).
pub(super) fn relate_type_and_user_type<'tcx>(
    infcx: &InferCtxt<'_, '_, 'tcx>,
    a: Ty<'tcx>,
    v: ty::Variance,
    b: CanonicalTy<'tcx>,
    locations: Locations,
    borrowck_context: Option<&mut BorrowCheckContext<'_, 'tcx>>,
) -> Fallible<()> {
    debug!(
        "sub_type_and_user_type(a={:?}, b={:?}, locations={:?})",
        a, b, locations
    );
    let Canonical {
        variables: b_variables,
        value: b_value,
    } = b;

    // The `TypeRelating` code assumes that the "canonical variables"
    // appear in the "a" side, so flip `Contravariant` ambient
    // variance to get the right relationship.
    let v1 = ty::Contravariant.xform(v);

    TypeRelating::new(
        infcx,
        v1,
        locations,
        borrowck_context,
        b_variables,
    ).relate(&b_value, &a)?;
    Ok(())
}

struct TypeRelating<'cx, 'bccx: 'cx, 'gcx: 'tcx, 'tcx: 'bccx> {
    infcx: &'cx InferCtxt<'cx, 'gcx, 'tcx>,

    /// How are we relating `a` and `b`?
    ///
    /// - covariant means `a <: b`
    /// - contravariant means `b <: a`
    /// - invariant means `a == b
    /// - bivariant means that it doesn't matter
    ambient_variance: ty::Variance,

    /// When we pass through a set of binders (e.g., when looking into
    /// a `fn` type), we push a new bound region scope onto here.  This
    /// will contain the instantiated region for each region in those
    /// binders. When we then encounter a `ReLateBound(d, br)`, we can
    /// use the debruijn index `d` to find the right scope, and then
    /// bound region name `br` to find the specific instantiation from
    /// within that scope. See `replace_bound_region`.
    ///
    /// This field stores the instantiations for late-bound regions in
    /// the `a` type.
    a_scopes: Vec<BoundRegionScope>,

    /// Same as `a_scopes`, but for the `b` type.
    b_scopes: Vec<BoundRegionScope>,

    /// Where (and why) is this relation taking place?
    locations: Locations,

    /// This will be `Some` when we are running the type check as part
    /// of NLL, and `None` if we are running a "sanity check".
    borrowck_context: Option<&'cx mut BorrowCheckContext<'bccx, 'tcx>>,

    /// As we execute, the type on the LHS *may* come from a canonical
    /// source. In that case, we will sometimes find a constraint like
    /// `?0 = B`, where `B` is a type from the RHS. The first time we
    /// find that, we simply record `B` (and the list of scopes that
    /// tells us how to *interpret* `B`). The next time we encounter
    /// `?0`, then, we can read this value out and use it.
    ///
    /// One problem: these variables may be in some other universe,
    /// how can we enforce that? I guess I could add some kind of
    /// "minimum universe constraint" that we can feed to the NLL checker.
    /// --> also, we know this doesn't happen
    canonical_var_values: IndexVec<CanonicalVar, Option<Kind<'tcx>>>,
}

#[derive(Clone, Debug)]
struct ScopesAndKind<'tcx> {
    scopes: Vec<BoundRegionScope>,
    kind: Kind<'tcx>,
}

#[derive(Clone, Debug, Default)]
struct BoundRegionScope {
    map: FxHashMap<ty::BoundRegion, RegionVid>,
}

#[derive(Copy, Clone)]
struct UniversallyQuantified(bool);

impl<'cx, 'bccx, 'gcx, 'tcx> TypeRelating<'cx, 'bccx, 'gcx, 'tcx> {
    fn new(
        infcx: &'cx InferCtxt<'cx, 'gcx, 'tcx>,
        ambient_variance: ty::Variance,
        locations: Locations,
        borrowck_context: Option<&'cx mut BorrowCheckContext<'bccx, 'tcx>>,
        canonical_var_infos: CanonicalVarInfos<'tcx>,
    ) -> Self {
        let canonical_var_values = IndexVec::from_elem_n(None, canonical_var_infos.len());
        Self {
            infcx,
            ambient_variance,
            borrowck_context,
            locations,
            canonical_var_values,
            a_scopes: vec![],
            b_scopes: vec![],
        }
    }

    fn ambient_covariance(&self) -> bool {
        match self.ambient_variance {
            ty::Variance::Covariant | ty::Variance::Invariant => true,
            ty::Variance::Contravariant | ty::Variance::Bivariant => false,
        }
    }

    fn ambient_contravariance(&self) -> bool {
        match self.ambient_variance {
            ty::Variance::Contravariant | ty::Variance::Invariant => true,
            ty::Variance::Covariant | ty::Variance::Bivariant => false,
        }
    }

    fn create_scope(
        &mut self,
        value: &ty::Binder<impl TypeFoldable<'tcx>>,
        universally_quantified: UniversallyQuantified,
    ) -> BoundRegionScope {
        let mut scope = BoundRegionScope::default();
        value.skip_binder().visit_with(&mut ScopeInstantiator {
            infcx: self.infcx,
            target_index: ty::INNERMOST,
            universally_quantified,
            bound_region_scope: &mut scope,
        });
        scope
    }

    /// When we encounter binders during the type traversal, we record
    /// the value to substitute for each of the things contained in
    /// that binder. (This will be either a universal placeholder or
    /// an existential inference variable.) Given the debruijn index
    /// `debruijn` (and name `br`) of some binder we have now
    /// encountered, this routine finds the value that we instantiated
    /// the region with; to do so, it indexes backwards into the list
    /// of ambient scopes `scopes`.
    fn lookup_bound_region(
        debruijn: ty::DebruijnIndex,
        br: &ty::BoundRegion,
        first_free_index: ty::DebruijnIndex,
        scopes: &[BoundRegionScope],
    ) -> RegionVid {
        // The debruijn index is a "reverse index" into the
        // scopes listing. So when we have INNERMOST (0), we
        // want the *last* scope pushed, and so forth.
        let debruijn_index = debruijn.index() - first_free_index.index();
        let scope = &scopes[scopes.len() - debruijn_index - 1];

        // Find this bound region in that scope to map to a
        // particular region.
        scope.map[br]
    }

    /// If `r` is a bound region, find the scope in which it is bound
    /// (from `scopes`) and return the value that we instantiated it
    /// with. Otherwise just return `r`.
    fn replace_bound_region(
        &self,
        universal_regions: &UniversalRegions<'tcx>,
        r: ty::Region<'tcx>,
        first_free_index: ty::DebruijnIndex,
        scopes: &[BoundRegionScope],
    ) -> RegionVid {
        match r {
            ty::ReLateBound(debruijn, br) => {
                Self::lookup_bound_region(*debruijn, br, first_free_index, scopes)
            }

            ty::ReVar(v) => *v,

            _ => universal_regions.to_region_vid(r),
        }
    }

    /// Push a new outlives requirement into our output set of
    /// constraints.
    fn push_outlives(&mut self, sup: RegionVid, sub: RegionVid) {
        debug!("push_outlives({:?}: {:?})", sup, sub);

        if let Some(borrowck_context) = &mut self.borrowck_context {
            borrowck_context
                .constraints
                .outlives_constraints
                .push(OutlivesConstraint {
                    sup,
                    sub,
                    locations: self.locations,
                });

            // FIXME all facts!
        }
    }

    /// When we encounter a canonical variable `var` in the output,
    /// equate it with `kind`. If the variable has been previously
    /// equated, then equate it again.
    fn relate_var(
        &mut self,
        var: CanonicalVar,
        b_kind: Kind<'tcx>,
    ) -> RelateResult<'tcx, Kind<'tcx>> {
        debug!("equate_var(var={:?}, b_kind={:?})", var, b_kind);

        let generalized_kind = match self.canonical_var_values[var] {
            Some(v) => v,
            None => {
                let generalized_kind = self.generalize_value(b_kind);
                self.canonical_var_values[var] = Some(generalized_kind);
                generalized_kind
            }
        };

        // The generalized values we extract from `canonical_var_values` have
        // been fully instantiated and hence the set of scopes we have
        // doesn't matter -- just to be sure, put an empty vector
        // in there.
        let old_a_scopes = ::std::mem::replace(&mut self.a_scopes, vec![]);

        // Relate the generalized kind to the original one.
        let result = self.relate(&generalized_kind, &b_kind);

        // Restore the old scopes now.
        self.a_scopes = old_a_scopes;

        debug!("equate_var: complete, result = {:?}", result);
        return result;
    }

    fn generalize_value(
        &self,
        kind: Kind<'tcx>,
    ) -> Kind<'tcx> {
        TypeGeneralizer {
            type_rel: self,
            first_free_index: ty::INNERMOST,
            ambient_variance: self.ambient_variance,

            // These always correspond to an `_` or `'_` written by
            // user, and those are always in the root universe.
            universe: ty::UniverseIndex::ROOT,
        }.relate(&kind, &kind)
            .unwrap()
    }
}

impl<'cx, 'bccx, 'gcx, 'tcx> TypeRelation<'cx, 'gcx, 'tcx>
    for TypeRelating<'cx, 'bccx, 'gcx, 'tcx>
{
    fn tcx(&self) -> TyCtxt<'cx, 'gcx, 'tcx> {
        self.infcx.tcx
    }

    fn tag(&self) -> &'static str {
        "nll::subtype"
    }

    fn a_is_expected(&self) -> bool {
        true
    }

    fn relate_with_variance<T: Relate<'tcx>>(
        &mut self,
        variance: ty::Variance,
        a: &T,
        b: &T,
    ) -> RelateResult<'tcx, T> {
        debug!(
            "relate_with_variance(variance={:?}, a={:?}, b={:?})",
            variance, a, b
        );

        let old_ambient_variance = self.ambient_variance;
        self.ambient_variance = self.ambient_variance.xform(variance);

        debug!(
            "relate_with_variance: ambient_variance = {:?}",
            self.ambient_variance
        );

        let r = self.relate(a, b)?;

        self.ambient_variance = old_ambient_variance;

        debug!("relate_with_variance: r={:?}", r);

        Ok(r)
    }

    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        // Watch out for the case that we are matching a `?T` against the
        // right-hand side.
        if let ty::Infer(ty::CanonicalTy(var)) = a.sty {
            self.relate_var(var, b.into())?;
            Ok(a)
        } else {
            debug!(
                "tys(a={:?}, b={:?}, variance={:?})",
                a, b, self.ambient_variance
            );

            relate::super_relate_tys(self, a, b)
        }
    }

    fn regions(
        &mut self,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        if let Some(&mut BorrowCheckContext {
            universal_regions, ..
        }) = self.borrowck_context
        {
            if let ty::ReCanonical(var) = a {
                self.relate_var(*var, b.into())?;
                return Ok(a);
            }

            debug!(
                "regions(a={:?}, b={:?}, variance={:?})",
                a, b, self.ambient_variance
            );

            let v_a =
                self.replace_bound_region(universal_regions, a, ty::INNERMOST, &self.a_scopes);
            let v_b =
                self.replace_bound_region(universal_regions, b, ty::INNERMOST, &self.b_scopes);

            debug!("regions: v_a = {:?}", v_a);
            debug!("regions: v_b = {:?}", v_b);

            if self.ambient_covariance() {
                // Covariance: a <= b. Hence, `b: a`.
                self.push_outlives(v_b, v_a);
            }

            if self.ambient_contravariance() {
                // Contravariant: b <= a. Hence, `a: b`.
                self.push_outlives(v_a, v_b);
            }
        }

        Ok(a)
    }

    fn binders<T>(
        &mut self,
        a: &ty::Binder<T>,
        b: &ty::Binder<T>,
    ) -> RelateResult<'tcx, ty::Binder<T>>
    where
        T: Relate<'tcx>,
    {
        // We want that
        //
        // ```
        // for<'a> fn(&'a u32) -> &'a u32 <:
        //   fn(&'b u32) -> &'b u32
        // ```
        //
        // but not
        //
        // ```
        // fn(&'a u32) -> &'a u32 <:
        //   for<'b> fn(&'b u32) -> &'b u32
        // ```
        //
        // We therefore proceed as follows:
        //
        // - Instantiate binders on `b` universally, yielding a universe U1.
        // - Instantiate binders on `a` existentially in U1.

        debug!(
            "binders({:?}: {:?}, ambient_variance={:?})",
            a, b, self.ambient_variance
        );

        if self.ambient_covariance() {
            // Covariance, so we want `for<..> A <: for<..> B` --
            // therefore we compare any instantiation of A (i.e., A
            // instantiated with existentials) against every
            // instantiation of B (i.e., B instantiated with
            // universals).

            let b_scope = self.create_scope(b, UniversallyQuantified(true));
            let a_scope = self.create_scope(a, UniversallyQuantified(false));

            debug!("binders: a_scope = {:?} (existential)", a_scope);
            debug!("binders: b_scope = {:?} (universal)", b_scope);

            self.b_scopes.push(b_scope);
            self.a_scopes.push(a_scope);

            // Reset the ambient variance to covariant. This is needed
            // to correctly handle cases like
            //
            //     for<'a> fn(&'a u32, &'a u3) == for<'b, 'c> fn(&'b u32, &'c u32)
            //
            // Somewhat surprisingly, these two types are actually
            // **equal**, even though the one on the right looks more
            // polymorphic. The reason is due to subtyping. To see it,
            // consider that each function can call the other:
            //
            // - The left function can call the right with `'b` and
            //   `'c` both equal to `'a`
            //
            // - The right function can call the left with `'a` set to
            //   `{P}`, where P is the point in the CFG where the call
            //   itself occurs. Note that `'b` and `'c` must both
            //   include P. At the point, the call works because of
            //   subtyping (i.e., `&'b u32 <: &{P} u32`).
            let variance = ::std::mem::replace(&mut self.ambient_variance, ty::Variance::Covariant);

            self.relate(a.skip_binder(), b.skip_binder())?;

            self.ambient_variance = variance;

            self.b_scopes.pop().unwrap();
            self.a_scopes.pop().unwrap();
        }

        if self.ambient_contravariance() {
            // Contravariance, so we want `for<..> A :> for<..> B`
            // -- therefore we compare every instantiation of A (i.e.,
            // A instantiated with universals) against any
            // instantiation of B (i.e., B instantiated with
            // existentials). Opposite of above.

            let a_scope = self.create_scope(a, UniversallyQuantified(true));
            let b_scope = self.create_scope(b, UniversallyQuantified(false));

            debug!("binders: a_scope = {:?} (universal)", a_scope);
            debug!("binders: b_scope = {:?} (existential)", b_scope);

            self.a_scopes.push(a_scope);
            self.b_scopes.push(b_scope);

            // Reset ambient variance to contravariance. See the
            // covariant case above for an explanation.
            let variance = ::std::mem::replace(
                &mut self.ambient_variance,
                ty::Variance::Contravariant,
            );

            self.relate(a.skip_binder(), b.skip_binder())?;

            self.ambient_variance = variance;

            self.b_scopes.pop().unwrap();
            self.a_scopes.pop().unwrap();
        }

        Ok(a.clone())
    }
}

/// When we encounter a binder like `for<..> fn(..)`, we actually have
/// to walk the `fn` value to find all the values bound by the `for`
/// (these are not explicitly present in the ty representation right
/// now). This visitor handles that: it descends the type, tracking
/// binder depth, and finds late-bound regions targeting the
/// `for<..`>.  For each of those, it creates an entry in
/// `bound_region_scope`.
struct ScopeInstantiator<'cx, 'gcx: 'tcx, 'tcx: 'cx> {
    infcx: &'cx InferCtxt<'cx, 'gcx, 'tcx>,
    // The debruijn index of the scope we are instantiating.
    target_index: ty::DebruijnIndex,
    universally_quantified: UniversallyQuantified,
    bound_region_scope: &'cx mut BoundRegionScope,
}

impl<'cx, 'gcx, 'tcx> TypeVisitor<'tcx> for ScopeInstantiator<'cx, 'gcx, 'tcx> {
    fn visit_binder<T: TypeFoldable<'tcx>>(&mut self, t: &ty::Binder<T>) -> bool {
        self.target_index.shift_in(1);
        t.super_visit_with(self);
        self.target_index.shift_out(1);

        false
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> bool {
        let ScopeInstantiator {
            infcx,
            universally_quantified,
            ..
        } = *self;

        match r {
            ty::ReLateBound(debruijn, br) if *debruijn == self.target_index => {
                self.bound_region_scope.map.entry(*br).or_insert_with(|| {
                    let origin = if universally_quantified.0 {
                        NLLRegionVariableOrigin::BoundRegion(infcx.create_subuniverse())
                    } else {
                        NLLRegionVariableOrigin::Existential
                    };
                    infcx.next_nll_region_var(origin).to_region_vid()
                });
            }

            _ => {}
        }

        false
    }
}

/// The "type generalize" is used when handling inference variables.
///
/// The basic strategy for handling a constraint like `?A <: B` is to
/// apply a "generalization strategy" to the type `B` -- this replaces
/// all the lifetimes in the type `B` with fresh inference
/// variables. (You can read more about the strategy in this [blog
/// post].)
///
/// As an example, if we had `?A <: &'x u32`, we would generalize `&'x
/// u32` to `&'0 u32` where `'0` is a fresh variable. This becomes the
/// value of `A`. Finally, we relate `&'0 u32 <: &'x u32`, which
/// establishes `'0: 'x` as a constraint.
///
/// As a side-effect of this generalization procedure, we also replace
/// all the bound regions that we have traversed with concrete values,
/// so that the resulting generalized type is independent from the
/// scopes.
///
/// [blog post]: https://is.gd/0hKvIr
struct TypeGeneralizer<'me, 'bccx: 'me, 'gcx: 'tcx, 'tcx: 'bccx> {
    type_rel: &'me TypeRelating<'me, 'bccx, 'gcx, 'tcx>,

    /// After we generalize this type, we are going to relative it to
    /// some other type. What will be the variance at this point?
    ambient_variance: ty::Variance,

    first_free_index: ty::DebruijnIndex,

    universe: ty::UniverseIndex,
}

impl TypeRelation<'me, 'gcx, 'tcx> for TypeGeneralizer<'me, 'bbcx, 'gcx, 'tcx> {
    fn tcx(&self) -> TyCtxt<'me, 'gcx, 'tcx> {
        self.type_rel.infcx.tcx
    }

    fn tag(&self) -> &'static str {
        "nll::generalizer"
    }

    fn a_is_expected(&self) -> bool {
        true
    }

    fn relate_with_variance<T: Relate<'tcx>>(
        &mut self,
        variance: ty::Variance,
        a: &T,
        b: &T,
    ) -> RelateResult<'tcx, T> {
        debug!(
            "TypeGeneralizer::relate_with_variance(variance={:?}, a={:?}, b={:?})",
            variance, a, b
        );

        let old_ambient_variance = self.ambient_variance;
        self.ambient_variance = self.ambient_variance.xform(variance);

        debug!(
            "TypeGeneralizer::relate_with_variance: ambient_variance = {:?}",
            self.ambient_variance
        );

        let r = self.relate(a, b)?;

        self.ambient_variance = old_ambient_variance;

        debug!("TypeGeneralizer::relate_with_variance: r={:?}", r);

        Ok(r)
    }

    fn tys(&mut self, a: Ty<'tcx>, _: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        debug!("TypeGeneralizer::tys(a={:?})", a,);

        match a.sty {
            ty::Infer(ty::TyVar(_)) | ty::Infer(ty::IntVar(_)) | ty::Infer(ty::FloatVar(_)) => {
                bug!(
                    "unexpected inference variable encountered in NLL generalization: {:?}",
                    a
                );
            }

            _ => relate::super_relate_tys(self, a, a),
        }
    }

    fn regions(
        &mut self,
        a: ty::Region<'tcx>,
        _: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        debug!("TypeGeneralizer::regions(a={:?})", a,);

        if let ty::ReLateBound(debruijn, _) = a {
            if *debruijn < self.first_free_index {
                return Ok(a);
            }
        }

        // For now, we just always create a fresh region variable to
        // replace all the regions in the source type. In the main
        // type checker, we special case the case where the ambient
        // variance is `Invariant` and try to avoid creating a fresh
        // region variable, but since this comes up so much less in
        // NLL (only when users use `_` etc) it is much less
        // important.
        //
        // As an aside, since these new variables are created in
        // `self.universe` universe, this also serves to enforce the
        // universe scoping rules.
        //
        // FIXME(#54105) -- if the ambient variance is bivariant,
        // though, we may however need to check well-formedness or
        // risk a problem like #41677 again.

        let replacement_region_vid = self.type_rel
            .infcx
            .next_nll_region_var_in_universe(NLLRegionVariableOrigin::Existential, self.universe);

        Ok(replacement_region_vid)
    }

    fn binders<T>(
        &mut self,
        a: &ty::Binder<T>,
        _: &ty::Binder<T>,
    ) -> RelateResult<'tcx, ty::Binder<T>>
    where
        T: Relate<'tcx>,
    {
        debug!("TypeGeneralizer::binders(a={:?})", a,);

        self.first_free_index.shift_in(1);
        let result = self.relate(a.skip_binder(), a.skip_binder())?;
        self.first_free_index.shift_out(1);
        Ok(ty::Binder::bind(result))
    }
}
