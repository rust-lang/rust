// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This code is kind of an alternate way of doing subtyping,
//! supertyping, and type equating, distinct from the `combine.rs`
//! code but very similar in its effect and design. Eventually the two
//! ought to be merged. This code is intended for use in NLL.
//!
//! Here are the key differences:
//!
//! - This code generally assumes that there are no unbound type
//!   inferences variables, because at NLL
//!   time types are fully inferred up-to regions.
//!   - Actually, to support user-given type annotations like
//!     `Vec<_>`, we do have some measure of support for type
//!     inference variables, but we impose some simplifying
//!     assumptions on them that would not be suitable for the infer
//!     code more generally. This could be fixed.
//! - This code uses "universes" to handle higher-ranked regions and
//!   not the leak-check. This is "more correct" than what rustc does
//!   and we are generally migrating in this direction, but NLL had to
//!   get there first.

use crate::infer::InferCtxt;
use crate::ty::fold::{TypeFoldable, TypeVisitor};
use crate::ty::relate::{self, Relate, RelateResult, TypeRelation};
use crate::ty::subst::Kind;
use crate::ty::{self, Ty, TyCtxt};
use rustc_data_structures::fx::FxHashMap;

pub struct TypeRelating<'me, 'gcx: 'tcx, 'tcx: 'me, D>
where
    D: TypeRelatingDelegate<'tcx>,
{
    infcx: &'me InferCtxt<'me, 'gcx, 'tcx>,

    /// Callback to use when we deduce an outlives relationship
    delegate: D,

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
    a_scopes: Vec<BoundRegionScope<'tcx>>,

    /// Same as `a_scopes`, but for the `b` type.
    b_scopes: Vec<BoundRegionScope<'tcx>>,
}

pub trait TypeRelatingDelegate<'tcx> {
    /// Push a constraint `sup: sub` -- this constraint must be
    /// satisfied for the two types to be related. `sub` and `sup` may
    /// be regions from the type or new variables created through the
    /// delegate.
    fn push_outlives(&mut self, sup: ty::Region<'tcx>, sub: ty::Region<'tcx>);

    /// Creates a new universe index. Used when instantiating placeholders.
    fn create_next_universe(&mut self) -> ty::UniverseIndex;

    /// Creates a new region variable representing a higher-ranked
    /// region that is instantiated existentially. This creates an
    /// inference variable, typically.
    ///
    /// So e.g. if you have `for<'a> fn(..) <: for<'b> fn(..)`, then
    /// we will invoke this method to instantiate `'a` with an
    /// inference variable (though `'b` would be instantiated first,
    /// as a placeholder).
    fn next_existential_region_var(&mut self) -> ty::Region<'tcx>;

    /// Creates a new region variable representing a
    /// higher-ranked region that is instantiated universally.
    /// This creates a new region placeholder, typically.
    ///
    /// So e.g. if you have `for<'a> fn(..) <: for<'b> fn(..)`, then
    /// we will invoke this method to instantiate `'b` with a
    /// placeholder region.
    fn next_placeholder_region(&mut self, placeholder: ty::PlaceholderRegion) -> ty::Region<'tcx>;

    /// Creates a new existential region in the given universe. This
    /// is used when handling subtyping and type variables -- if we
    /// have that `?X <: Foo<'a>`, for example, we would instantiate
    /// `?X` with a type like `Foo<'?0>` where `'?0` is a fresh
    /// existential variable created by this function. We would then
    /// relate `Foo<'?0>` with `Foo<'a>` (and probably add an outlives
    /// relation stating that `'?0: 'a`).
    fn generalize_existential(&mut self, universe: ty::UniverseIndex) -> ty::Region<'tcx>;
}

#[derive(Clone, Debug)]
struct ScopesAndKind<'tcx> {
    scopes: Vec<BoundRegionScope<'tcx>>,
    kind: Kind<'tcx>,
}

#[derive(Clone, Debug, Default)]
struct BoundRegionScope<'tcx> {
    map: FxHashMap<ty::BoundRegion, ty::Region<'tcx>>,
}

#[derive(Copy, Clone)]
struct UniversallyQuantified(bool);

impl<'me, 'gcx, 'tcx, D> TypeRelating<'me, 'gcx, 'tcx, D>
where
    D: TypeRelatingDelegate<'tcx>,
{
    pub fn new(
        infcx: &'me InferCtxt<'me, 'gcx, 'tcx>,
        delegate: D,
        ambient_variance: ty::Variance,
    ) -> Self {
        Self {
            infcx,
            delegate,
            ambient_variance,
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
    ) -> BoundRegionScope<'tcx> {
        let mut scope = BoundRegionScope::default();

        // Create a callback that creates (via the delegate) either an
        // existential or placeholder region as needed.
        let mut next_region = {
            let delegate = &mut self.delegate;
            let mut lazy_universe = None;
            move |br: ty::BoundRegion| {
                if universally_quantified.0 {
                    // The first time this closure is called, create a
                    // new universe for the placeholders we will make
                    // from here out.
                    let universe = lazy_universe.unwrap_or_else(|| {
                        let universe = delegate.create_next_universe();
                        lazy_universe = Some(universe);
                        universe
                    });

                    let placeholder = ty::PlaceholderRegion { universe, name: br };
                    delegate.next_placeholder_region(placeholder)
                } else {
                    delegate.next_existential_region_var()
                }
            }
        };

        value.skip_binder().visit_with(&mut ScopeInstantiator {
            next_region: &mut next_region,
            target_index: ty::INNERMOST,
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
        scopes: &[BoundRegionScope<'tcx>],
    ) -> ty::Region<'tcx> {
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
        r: ty::Region<'tcx>,
        first_free_index: ty::DebruijnIndex,
        scopes: &[BoundRegionScope<'tcx>],
    ) -> ty::Region<'tcx> {
        if let ty::ReLateBound(debruijn, br) = r {
            Self::lookup_bound_region(*debruijn, br, first_free_index, scopes)
        } else {
            r
        }
    }

    /// Push a new outlives requirement into our output set of
    /// constraints.
    fn push_outlives(&mut self, sup: ty::Region<'tcx>, sub: ty::Region<'tcx>) {
        debug!("push_outlives({:?}: {:?})", sup, sub);

        self.delegate.push_outlives(sup, sub);
    }

    /// When we encounter a canonical variable `var` in the output,
    /// equate it with `kind`. If the variable has been previously
    /// equated, then equate it again.
    fn relate_var(&mut self, var_ty: Ty<'tcx>, value_ty: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        debug!("equate_var(var_ty={:?}, value_ty={:?})", var_ty, value_ty);

        let generalized_ty = self.generalize_value(value_ty);
        self.infcx
            .force_instantiate_unchecked(var_ty, generalized_ty);

        // The generalized values we extract from `canonical_var_values` have
        // been fully instantiated and hence the set of scopes we have
        // doesn't matter -- just to be sure, put an empty vector
        // in there.
        let old_a_scopes = ::std::mem::replace(&mut self.a_scopes, vec![]);

        // Relate the generalized kind to the original one.
        let result = self.relate(&generalized_ty, &value_ty);

        // Restore the old scopes now.
        self.a_scopes = old_a_scopes;

        debug!("equate_var: complete, result = {:?}", result);
        result
    }

    fn generalize_value<T: Relate<'tcx>>(&mut self, value: T) -> T {
        TypeGeneralizer {
            tcx: self.infcx.tcx,
            delegate: &mut self.delegate,
            first_free_index: ty::INNERMOST,
            ambient_variance: self.ambient_variance,

            // These always correspond to an `_` or `'_` written by
            // user, and those are always in the root universe.
            universe: ty::UniverseIndex::ROOT,
        }.relate(&value, &value)
            .unwrap()
    }
}

impl<D> TypeRelation<'me, 'gcx, 'tcx> for TypeRelating<'me, 'gcx, 'tcx, D>
where
    D: TypeRelatingDelegate<'tcx>,
{
    fn tcx(&self) -> TyCtxt<'me, 'gcx, 'tcx> {
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
        let a = self.infcx.shallow_resolve(a);
        match a.sty {
            ty::Infer(ty::TyVar(_)) | ty::Infer(ty::IntVar(_)) | ty::Infer(ty::FloatVar(_)) => {
                self.relate_var(a.into(), b.into())
            }

            _ => {
                debug!(
                    "tys(a={:?}, b={:?}, variance={:?})",
                    a, b, self.ambient_variance
                );

                relate::super_relate_tys(self, a, b)
            }
        }
    }

    fn regions(
        &mut self,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        debug!(
            "regions(a={:?}, b={:?}, variance={:?})",
            a, b, self.ambient_variance
        );

        let v_a = self.replace_bound_region(a, ty::INNERMOST, &self.a_scopes);
        let v_b = self.replace_bound_region(b, ty::INNERMOST, &self.b_scopes);

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
            let variance =
                ::std::mem::replace(&mut self.ambient_variance, ty::Variance::Contravariant);

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
struct ScopeInstantiator<'me, 'tcx: 'me> {
    next_region: &'me mut dyn FnMut(ty::BoundRegion) -> ty::Region<'tcx>,
    // The debruijn index of the scope we are instantiating.
    target_index: ty::DebruijnIndex,
    bound_region_scope: &'me mut BoundRegionScope<'tcx>,
}

impl<'me, 'tcx> TypeVisitor<'tcx> for ScopeInstantiator<'me, 'tcx> {
    fn visit_binder<T: TypeFoldable<'tcx>>(&mut self, t: &ty::Binder<T>) -> bool {
        self.target_index.shift_in(1);
        t.super_visit_with(self);
        self.target_index.shift_out(1);

        false
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> bool {
        let ScopeInstantiator {
            bound_region_scope,
            next_region,
            ..
        } = self;

        match r {
            ty::ReLateBound(debruijn, br) if *debruijn == self.target_index => {
                bound_region_scope
                    .map
                    .entry(*br)
                    .or_insert_with(|| next_region(*br));
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
struct TypeGeneralizer<'me, 'gcx: 'tcx, 'tcx: 'me, D>
where
    D: TypeRelatingDelegate<'tcx> + 'me,
{
    tcx: TyCtxt<'me, 'gcx, 'tcx>,

    delegate: &'me mut D,

    /// After we generalize this type, we are going to relative it to
    /// some other type. What will be the variance at this point?
    ambient_variance: ty::Variance,

    first_free_index: ty::DebruijnIndex,

    universe: ty::UniverseIndex,
}

impl<D> TypeRelation<'me, 'gcx, 'tcx> for TypeGeneralizer<'me, 'gcx, 'tcx, D>
where
    D: TypeRelatingDelegate<'tcx>,
{
    fn tcx(&self) -> TyCtxt<'me, 'gcx, 'tcx> {
        self.tcx
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

        let replacement_region_vid = self.delegate.generalize_existential(self.universe);

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

impl InferCtxt<'_, '_, 'tcx> {
    /// A hacky sort of method used by the NLL type-relating code:
    ///
    /// - `var` must be some unbound type variable.
    /// - `value` must be a suitable type to use as its value.
    ///
    /// `var` will then be equated with `value`. Note that this
    /// sidesteps a number of important checks, such as the "occurs
    /// check" that prevents cyclic types, so it is important not to
    /// use this method during regular type-check.
    fn force_instantiate_unchecked(&self, var: Ty<'tcx>, value: Ty<'tcx>) {
        match (&var.sty, &value.sty) {
            (&ty::Infer(ty::TyVar(vid)), _) => {
                let mut type_variables = self.type_variables.borrow_mut();

                // In NLL, we don't have type inference variables
                // floating around, so we can do this rather imprecise
                // variant of the occurs-check.
                assert!(!value.has_infer_types());

                type_variables.instantiate(vid, value);
            }

            (&ty::Infer(ty::IntVar(vid)), &ty::Int(value)) => {
                let mut int_unification_table = self.int_unification_table.borrow_mut();
                int_unification_table
                    .unify_var_value(vid, Some(ty::IntVarValue::IntType(value)))
                    .unwrap_or_else(|_| {
                        bug!("failed to unify int var `{:?}` with `{:?}`", vid, value);
                    });
            }

            (&ty::Infer(ty::IntVar(vid)), &ty::Uint(value)) => {
                let mut int_unification_table = self.int_unification_table.borrow_mut();
                int_unification_table
                    .unify_var_value(vid, Some(ty::IntVarValue::UintType(value)))
                    .unwrap_or_else(|_| {
                        bug!("failed to unify int var `{:?}` with `{:?}`", vid, value);
                    });
            }

            (&ty::Infer(ty::FloatVar(vid)), &ty::Float(value)) => {
                let mut float_unification_table = self.float_unification_table.borrow_mut();
                float_unification_table
                    .unify_var_value(vid, Some(ty::FloatVarValue(value)))
                    .unwrap_or_else(|_| {
                        bug!("failed to unify float var `{:?}` with `{:?}`", vid, value)
                    });
            }

            _ => {
                bug!(
                    "force_instantiate_unchecked invoked with bad combination: var={:?} value={:?}",
                    var,
                    value,
                );
            }
        }
    }
}
