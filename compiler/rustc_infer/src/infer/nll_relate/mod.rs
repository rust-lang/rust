//! This code is kind of an alternate way of doing subtyping,
//! supertyping, and type equating, distinct from the `combine.rs`
//! code but very similar in its effect and design. Eventually the two
//! ought to be merged. This code is intended for use in NLL and chalk.
//!
//! Here are the key differences:
//!
//! - This code may choose to bypass some checks (e.g., the occurs check)
//!   in the case where we know that there are no unbound type inference
//!   variables. This is the case for NLL, because at NLL time types are fully
//!   inferred up-to regions.
//! - This code uses "universes" to handle higher-ranked regions and
//!   not the leak-check. This is "more correct" than what rustc does
//!   and we are generally migrating in this direction, but NLL had to
//!   get there first.
//!
//! Also, this code assumes that there are no bound types at all, not even
//! free ones. This is ok because:
//! - we are not relating anything quantified over some type variable
//! - we will have instantiated all the bound type vars already (the one
//!   thing we relate in chalk are basically domain goals and their
//!   constituents)

use crate::infer::combine::ConstEquateRelation;
use crate::infer::InferCtxt;
use crate::infer::{ConstVarValue, ConstVariableValue};
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::fold::{TypeFoldable, TypeVisitor};
use rustc_middle::ty::relate::{self, Relate, RelateResult, TypeRelation};
use rustc_middle::ty::{self, InferConst, Ty, TyCtxt};
use std::fmt::Debug;
use std::ops::ControlFlow;

#[derive(PartialEq)]
pub enum NormalizationStrategy {
    Lazy,
    Eager,
}

pub struct TypeRelating<'me, 'tcx, D>
where
    D: TypeRelatingDelegate<'tcx>,
{
    infcx: &'me InferCtxt<'me, 'tcx>,

    /// Callback to use when we deduce an outlives relationship
    delegate: D,

    /// How are we relating `a` and `b`?
    ///
    /// - Covariant means `a <: b`.
    /// - Contravariant means `b <: a`.
    /// - Invariant means `a == b.
    /// - Bivariant means that it doesn't matter.
    ambient_variance: ty::Variance,

    /// When we pass through a set of binders (e.g., when looking into
    /// a `fn` type), we push a new bound region scope onto here. This
    /// will contain the instantiated region for each region in those
    /// binders. When we then encounter a `ReLateBound(d, br)`, we can
    /// use the De Bruijn index `d` to find the right scope, and then
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

    fn const_equate(&mut self, a: &'tcx ty::Const<'tcx>, b: &'tcx ty::Const<'tcx>);

    /// Creates a new universe index. Used when instantiating placeholders.
    fn create_next_universe(&mut self) -> ty::UniverseIndex;

    /// Creates a new region variable representing a higher-ranked
    /// region that is instantiated existentially. This creates an
    /// inference variable, typically.
    ///
    /// So e.g., if you have `for<'a> fn(..) <: for<'b> fn(..)`, then
    /// we will invoke this method to instantiate `'a` with an
    /// inference variable (though `'b` would be instantiated first,
    /// as a placeholder).
    fn next_existential_region_var(&mut self, was_placeholder: bool) -> ty::Region<'tcx>;

    /// Creates a new region variable representing a
    /// higher-ranked region that is instantiated universally.
    /// This creates a new region placeholder, typically.
    ///
    /// So e.g., if you have `for<'a> fn(..) <: for<'b> fn(..)`, then
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

    /// Define the normalization strategy to use, eager or lazy.
    fn normalization() -> NormalizationStrategy;

    /// Enables some optimizations if we do not expect inference variables
    /// in the RHS of the relation.
    fn forbid_inference_vars() -> bool;
}

#[derive(Clone, Debug, Default)]
struct BoundRegionScope<'tcx> {
    map: FxHashMap<ty::BoundRegion, ty::Region<'tcx>>,
}

#[derive(Copy, Clone)]
struct UniversallyQuantified(bool);

impl<'me, 'tcx, D> TypeRelating<'me, 'tcx, D>
where
    D: TypeRelatingDelegate<'tcx>,
{
    pub fn new(
        infcx: &'me InferCtxt<'me, 'tcx>,
        delegate: D,
        ambient_variance: ty::Variance,
    ) -> Self {
        Self { infcx, delegate, ambient_variance, a_scopes: vec![], b_scopes: vec![] }
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
        value: ty::Binder<impl Relate<'tcx>>,
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

                    let placeholder = ty::PlaceholderRegion { universe, name: br.kind };
                    delegate.next_placeholder_region(placeholder)
                } else {
                    delegate.next_existential_region_var(true)
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
    /// an existential inference variable.) Given the De Bruijn index
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
        debug!("replace_bound_regions(scopes={:?})", scopes);
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

    /// Relate a projection type and some value type lazily. This will always
    /// succeed, but we push an additional `ProjectionEq` goal depending
    /// on the value type:
    /// - if the value type is any type `T` which is not a projection, we push
    ///   `ProjectionEq(projection = T)`.
    /// - if the value type is another projection `other_projection`, we create
    ///   a new inference variable `?U` and push the two goals
    ///   `ProjectionEq(projection = ?U)`, `ProjectionEq(other_projection = ?U)`.
    fn relate_projection_ty(
        &mut self,
        projection_ty: ty::ProjectionTy<'tcx>,
        value_ty: Ty<'tcx>,
    ) -> Ty<'tcx> {
        use crate::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
        use rustc_span::DUMMY_SP;

        match *value_ty.kind() {
            ty::Projection(other_projection_ty) => {
                let var = self.infcx.next_ty_var(TypeVariableOrigin {
                    kind: TypeVariableOriginKind::MiscVariable,
                    span: DUMMY_SP,
                });
                self.relate_projection_ty(projection_ty, var);
                self.relate_projection_ty(other_projection_ty, var);
                var
            }

            _ => bug!("should never be invoked with eager normalization"),
        }
    }

    /// Relate a type inference variable with a value type. This works
    /// by creating a "generalization" G of the value where all the
    /// lifetimes are replaced with fresh inference values. This
    /// genearlization G becomes the value of the inference variable,
    /// and is then related in turn to the value. So e.g. if you had
    /// `vid = ?0` and `value = &'a u32`, we might first instantiate
    /// `?0` to a type like `&'0 u32` where `'0` is a fresh variable,
    /// and then relate `&'0 u32` with `&'a u32` (resulting in
    /// relations between `'0` and `'a`).
    ///
    /// The variable `pair` can be either a `(vid, ty)` or `(ty, vid)`
    /// -- in other words, it is always a (unresolved) inference
    /// variable `vid` and a type `ty` that are being related, but the
    /// vid may appear either as the "a" type or the "b" type,
    /// depending on where it appears in the tuple. The trait
    /// `VidValuePair` lets us work with the vid/type while preserving
    /// the "sidedness" when necessary -- the sidedness is relevant in
    /// particular for the variance and set of in-scope things.
    fn relate_ty_var<PAIR: VidValuePair<'tcx>>(
        &mut self,
        pair: PAIR,
    ) -> RelateResult<'tcx, Ty<'tcx>> {
        debug!("relate_ty_var({:?})", pair);

        let vid = pair.vid();
        let value_ty = pair.value_ty();

        // FIXME(invariance) -- this logic assumes invariance, but that is wrong.
        // This only presently applies to chalk integration, as NLL
        // doesn't permit type variables to appear on both sides (and
        // doesn't use lazy norm).
        match *value_ty.kind() {
            ty::Infer(ty::TyVar(value_vid)) => {
                // Two type variables: just equate them.
                self.infcx.inner.borrow_mut().type_variables().equate(vid, value_vid);
                return Ok(value_ty);
            }

            ty::Projection(projection_ty) if D::normalization() == NormalizationStrategy::Lazy => {
                return Ok(self.relate_projection_ty(projection_ty, self.infcx.tcx.mk_ty_var(vid)));
            }

            _ => (),
        }

        let generalized_ty = self.generalize_value(value_ty, vid)?;
        debug!("relate_ty_var: generalized_ty = {:?}", generalized_ty);

        if D::forbid_inference_vars() {
            // In NLL, we don't have type inference variables
            // floating around, so we can do this rather imprecise
            // variant of the occurs-check.
            assert!(!generalized_ty.has_infer_types_or_consts());
        }

        self.infcx.inner.borrow_mut().type_variables().instantiate(vid, generalized_ty);

        // The generalized values we extract from `canonical_var_values` have
        // been fully instantiated and hence the set of scopes we have
        // doesn't matter -- just to be sure, put an empty vector
        // in there.
        let old_a_scopes = std::mem::take(pair.vid_scopes(self));

        // Relate the generalized kind to the original one.
        let result = pair.relate_generalized_ty(self, generalized_ty);

        // Restore the old scopes now.
        *pair.vid_scopes(self) = old_a_scopes;

        debug!("relate_ty_var: complete, result = {:?}", result);
        result
    }

    fn generalize_value<T: Relate<'tcx>>(
        &mut self,
        value: T,
        for_vid: ty::TyVid,
    ) -> RelateResult<'tcx, T> {
        let universe = self.infcx.probe_ty_var(for_vid).unwrap_err();

        let mut generalizer = TypeGeneralizer {
            infcx: self.infcx,
            delegate: &mut self.delegate,
            first_free_index: ty::INNERMOST,
            ambient_variance: self.ambient_variance,
            for_vid_sub_root: self.infcx.inner.borrow_mut().type_variables().sub_root_var(for_vid),
            universe,
        };

        generalizer.relate(value, value)
    }
}

/// When we instantiate a inference variable with a value in
/// `relate_ty_var`, we always have the pair of a `TyVid` and a `Ty`,
/// but the ordering may vary (depending on whether the inference
/// variable was found on the `a` or `b` sides). Therefore, this trait
/// allows us to factor out common code, while preserving the order
/// when needed.
trait VidValuePair<'tcx>: Debug {
    /// Extract the inference variable (which could be either the
    /// first or second part of the tuple).
    fn vid(&self) -> ty::TyVid;

    /// Extract the value it is being related to (which will be the
    /// opposite part of the tuple from the vid).
    fn value_ty(&self) -> Ty<'tcx>;

    /// Extract the scopes that apply to whichever side of the tuple
    /// the vid was found on.  See the comment where this is called
    /// for more details on why we want them.
    fn vid_scopes<D: TypeRelatingDelegate<'tcx>>(
        &self,
        relate: &'r mut TypeRelating<'_, 'tcx, D>,
    ) -> &'r mut Vec<BoundRegionScope<'tcx>>;

    /// Given a generalized type G that should replace the vid, relate
    /// G to the value, putting G on whichever side the vid would have
    /// appeared.
    fn relate_generalized_ty<D>(
        &self,
        relate: &mut TypeRelating<'_, 'tcx, D>,
        generalized_ty: Ty<'tcx>,
    ) -> RelateResult<'tcx, Ty<'tcx>>
    where
        D: TypeRelatingDelegate<'tcx>;
}

impl VidValuePair<'tcx> for (ty::TyVid, Ty<'tcx>) {
    fn vid(&self) -> ty::TyVid {
        self.0
    }

    fn value_ty(&self) -> Ty<'tcx> {
        self.1
    }

    fn vid_scopes<D>(
        &self,
        relate: &'r mut TypeRelating<'_, 'tcx, D>,
    ) -> &'r mut Vec<BoundRegionScope<'tcx>>
    where
        D: TypeRelatingDelegate<'tcx>,
    {
        &mut relate.a_scopes
    }

    fn relate_generalized_ty<D>(
        &self,
        relate: &mut TypeRelating<'_, 'tcx, D>,
        generalized_ty: Ty<'tcx>,
    ) -> RelateResult<'tcx, Ty<'tcx>>
    where
        D: TypeRelatingDelegate<'tcx>,
    {
        relate.relate(&generalized_ty, &self.value_ty())
    }
}

// In this case, the "vid" is the "b" type.
impl VidValuePair<'tcx> for (Ty<'tcx>, ty::TyVid) {
    fn vid(&self) -> ty::TyVid {
        self.1
    }

    fn value_ty(&self) -> Ty<'tcx> {
        self.0
    }

    fn vid_scopes<D>(
        &self,
        relate: &'r mut TypeRelating<'_, 'tcx, D>,
    ) -> &'r mut Vec<BoundRegionScope<'tcx>>
    where
        D: TypeRelatingDelegate<'tcx>,
    {
        &mut relate.b_scopes
    }

    fn relate_generalized_ty<D>(
        &self,
        relate: &mut TypeRelating<'_, 'tcx, D>,
        generalized_ty: Ty<'tcx>,
    ) -> RelateResult<'tcx, Ty<'tcx>>
    where
        D: TypeRelatingDelegate<'tcx>,
    {
        relate.relate(&self.value_ty(), &generalized_ty)
    }
}

impl<D> TypeRelation<'tcx> for TypeRelating<'me, 'tcx, D>
where
    D: TypeRelatingDelegate<'tcx>,
{
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    // FIXME(oli-obk): not sure how to get the correct ParamEnv
    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        ty::ParamEnv::empty()
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
        a: T,
        b: T,
    ) -> RelateResult<'tcx, T> {
        debug!("relate_with_variance(variance={:?}, a={:?}, b={:?})", variance, a, b);

        let old_ambient_variance = self.ambient_variance;
        self.ambient_variance = self.ambient_variance.xform(variance);

        debug!("relate_with_variance: ambient_variance = {:?}", self.ambient_variance);

        let r = self.relate(a, b)?;

        self.ambient_variance = old_ambient_variance;

        debug!("relate_with_variance: r={:?}", r);

        Ok(r)
    }

    fn tys(&mut self, a: Ty<'tcx>, mut b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        let a = self.infcx.shallow_resolve(a);

        if !D::forbid_inference_vars() {
            b = self.infcx.shallow_resolve(b);
        }

        if a == b {
            // Subtle: if a or b has a bound variable that we are lazilly
            // substituting, then even if a == b, it could be that the values we
            // will substitute for those bound variables are *not* the same, and
            // hence returning `Ok(a)` is incorrect.
            if !a.has_escaping_bound_vars() && !b.has_escaping_bound_vars() {
                return Ok(a);
            }
        }

        match (a.kind(), b.kind()) {
            (_, &ty::Infer(ty::TyVar(vid))) => {
                if D::forbid_inference_vars() {
                    // Forbid inference variables in the RHS.
                    bug!("unexpected inference var {:?}", b)
                } else {
                    self.relate_ty_var((a, vid))
                }
            }

            (&ty::Infer(ty::TyVar(vid)), _) => self.relate_ty_var((vid, b)),

            (&ty::Projection(projection_ty), _)
                if D::normalization() == NormalizationStrategy::Lazy =>
            {
                Ok(self.relate_projection_ty(projection_ty, b))
            }

            (_, &ty::Projection(projection_ty))
                if D::normalization() == NormalizationStrategy::Lazy =>
            {
                Ok(self.relate_projection_ty(projection_ty, a))
            }

            _ => {
                debug!("tys(a={:?}, b={:?}, variance={:?})", a, b, self.ambient_variance);

                // Will also handle unification of `IntVar` and `FloatVar`.
                self.infcx.super_combine_tys(self, a, b)
            }
        }
    }

    fn regions(
        &mut self,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        debug!("regions(a={:?}, b={:?}, variance={:?})", a, b, self.ambient_variance);

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

    fn consts(
        &mut self,
        a: &'tcx ty::Const<'tcx>,
        mut b: &'tcx ty::Const<'tcx>,
    ) -> RelateResult<'tcx, &'tcx ty::Const<'tcx>> {
        let a = self.infcx.shallow_resolve(a);

        if !D::forbid_inference_vars() {
            b = self.infcx.shallow_resolve(b);
        }

        match b.val {
            ty::ConstKind::Infer(InferConst::Var(_)) if D::forbid_inference_vars() => {
                // Forbid inference variables in the RHS.
                bug!("unexpected inference var {:?}", b)
            }
            // FIXME(invariance): see the related FIXME above.
            _ => self.infcx.super_combine_consts(self, a, b),
        }
    }

    fn binders<T>(
        &mut self,
        a: ty::Binder<T>,
        b: ty::Binder<T>,
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

        debug!("binders({:?}: {:?}, ambient_variance={:?})", a, b, self.ambient_variance);

        if let (Some(a), Some(b)) = (a.no_bound_vars(), b.no_bound_vars()) {
            // Fast path for the common case.
            self.relate(a, b)?;
            return Ok(ty::Binder::dummy(a));
        }

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
            //     for<'a> fn(&'a u32, &'a u32) == for<'b, 'c> fn(&'b u32, &'c u32)
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
            let variance = std::mem::replace(&mut self.ambient_variance, ty::Variance::Covariant);

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
                std::mem::replace(&mut self.ambient_variance, ty::Variance::Contravariant);

            self.relate(a.skip_binder(), b.skip_binder())?;

            self.ambient_variance = variance;

            self.b_scopes.pop().unwrap();
            self.a_scopes.pop().unwrap();
        }

        Ok(a)
    }
}

impl<'tcx, D> ConstEquateRelation<'tcx> for TypeRelating<'_, 'tcx, D>
where
    D: TypeRelatingDelegate<'tcx>,
{
    fn const_equate_obligation(&mut self, a: &'tcx ty::Const<'tcx>, b: &'tcx ty::Const<'tcx>) {
        self.delegate.const_equate(a, b);
    }
}

/// When we encounter a binder like `for<..> fn(..)`, we actually have
/// to walk the `fn` value to find all the values bound by the `for`
/// (these are not explicitly present in the ty representation right
/// now). This visitor handles that: it descends the type, tracking
/// binder depth, and finds late-bound regions targeting the
/// `for<..`>.  For each of those, it creates an entry in
/// `bound_region_scope`.
struct ScopeInstantiator<'me, 'tcx> {
    next_region: &'me mut dyn FnMut(ty::BoundRegion) -> ty::Region<'tcx>,
    // The debruijn index of the scope we are instantiating.
    target_index: ty::DebruijnIndex,
    bound_region_scope: &'me mut BoundRegionScope<'tcx>,
}

impl<'me, 'tcx> TypeVisitor<'tcx> for ScopeInstantiator<'me, 'tcx> {
    fn visit_binder<T: TypeFoldable<'tcx>>(
        &mut self,
        t: &ty::Binder<T>,
    ) -> ControlFlow<Self::BreakTy> {
        self.target_index.shift_in(1);
        t.super_visit_with(self);
        self.target_index.shift_out(1);

        ControlFlow::CONTINUE
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
        let ScopeInstantiator { bound_region_scope, next_region, .. } = self;

        match r {
            ty::ReLateBound(debruijn, br) if *debruijn == self.target_index => {
                bound_region_scope.map.entry(*br).or_insert_with(|| next_region(*br));
            }

            _ => {}
        }

        ControlFlow::CONTINUE
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
struct TypeGeneralizer<'me, 'tcx, D>
where
    D: TypeRelatingDelegate<'tcx>,
{
    infcx: &'me InferCtxt<'me, 'tcx>,

    delegate: &'me mut D,

    /// After we generalize this type, we are going to relative it to
    /// some other type. What will be the variance at this point?
    ambient_variance: ty::Variance,

    first_free_index: ty::DebruijnIndex,

    /// The vid of the type variable that is in the process of being
    /// instantiated. If we find this within the value we are folding,
    /// that means we would have created a cyclic value.
    for_vid_sub_root: ty::TyVid,

    /// The universe of the type variable that is in the process of being
    /// instantiated. If we find anything that this universe cannot name,
    /// we reject the relation.
    universe: ty::UniverseIndex,
}

impl<D> TypeRelation<'tcx> for TypeGeneralizer<'me, 'tcx, D>
where
    D: TypeRelatingDelegate<'tcx>,
{
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    // FIXME(oli-obk): not sure how to get the correct ParamEnv
    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        ty::ParamEnv::empty()
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
        a: T,
        b: T,
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
        use crate::infer::type_variable::TypeVariableValue;

        debug!("TypeGeneralizer::tys(a={:?})", a);

        match *a.kind() {
            ty::Infer(ty::TyVar(_)) | ty::Infer(ty::IntVar(_)) | ty::Infer(ty::FloatVar(_))
                if D::forbid_inference_vars() =>
            {
                bug!("unexpected inference variable encountered in NLL generalization: {:?}", a);
            }

            ty::Infer(ty::TyVar(vid)) => {
                let mut inner = self.infcx.inner.borrow_mut();
                let variables = &mut inner.type_variables();
                let vid = variables.root_var(vid);
                let sub_vid = variables.sub_root_var(vid);
                if sub_vid == self.for_vid_sub_root {
                    // If sub-roots are equal, then `for_vid` and
                    // `vid` are related via subtyping.
                    debug!("TypeGeneralizer::tys: occurs check failed");
                    Err(TypeError::Mismatch)
                } else {
                    match variables.probe(vid) {
                        TypeVariableValue::Known { value: u } => {
                            drop(variables);
                            self.relate(u, u)
                        }
                        TypeVariableValue::Unknown { universe: _universe } => {
                            if self.ambient_variance == ty::Bivariant {
                                // FIXME: we may need a WF predicate (related to #54105).
                            }

                            let origin = *variables.var_origin(vid);

                            // Replacing with a new variable in the universe `self.universe`,
                            // it will be unified later with the original type variable in
                            // the universe `_universe`.
                            let new_var_id = variables.new_var(self.universe, false, origin);

                            let u = self.tcx().mk_ty_var(new_var_id);
                            debug!("generalize: replacing original vid={:?} with new={:?}", vid, u);
                            Ok(u)
                        }
                    }
                }
            }

            ty::Infer(ty::IntVar(_) | ty::FloatVar(_)) => {
                // No matter what mode we are in,
                // integer/floating-point types must be equal to be
                // relatable.
                Ok(a)
            }

            ty::Placeholder(placeholder) => {
                if self.universe.cannot_name(placeholder.universe) {
                    debug!(
                        "TypeGeneralizer::tys: root universe {:?} cannot name\
                         placeholder in universe {:?}",
                        self.universe, placeholder.universe
                    );
                    Err(TypeError::Mismatch)
                } else {
                    Ok(a)
                }
            }

            _ => relate::super_relate_tys(self, a, a),
        }
    }

    fn regions(
        &mut self,
        a: ty::Region<'tcx>,
        _: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        debug!("TypeGeneralizer::regions(a={:?})", a);

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

    fn consts(
        &mut self,
        a: &'tcx ty::Const<'tcx>,
        _: &'tcx ty::Const<'tcx>,
    ) -> RelateResult<'tcx, &'tcx ty::Const<'tcx>> {
        match a.val {
            ty::ConstKind::Infer(InferConst::Var(_)) if D::forbid_inference_vars() => {
                bug!("unexpected inference variable encountered in NLL generalization: {:?}", a);
            }
            ty::ConstKind::Infer(InferConst::Var(vid)) => {
                let mut inner = self.infcx.inner.borrow_mut();
                let variable_table = &mut inner.const_unification_table();
                let var_value = variable_table.probe_value(vid);
                match var_value.val.known() {
                    Some(u) => self.relate(u, u),
                    None => {
                        let new_var_id = variable_table.new_key(ConstVarValue {
                            origin: var_value.origin,
                            val: ConstVariableValue::Unknown { universe: self.universe },
                        });
                        Ok(self.tcx().mk_const_var(new_var_id, a.ty))
                    }
                }
            }
            ty::ConstKind::Unevaluated(..) if self.tcx().lazy_normalization() => Ok(a),
            _ => relate::super_relate_consts(self, a, a),
        }
    }

    fn binders<T>(
        &mut self,
        a: ty::Binder<T>,
        _: ty::Binder<T>,
    ) -> RelateResult<'tcx, ty::Binder<T>>
    where
        T: Relate<'tcx>,
    {
        debug!("TypeGeneralizer::binders(a={:?})", a);

        self.first_free_index.shift_in(1);
        let result = self.relate(a.skip_binder(), a.skip_binder())?;
        self.first_free_index.shift_out(1);
        Ok(a.rebind(result))
    }
}
