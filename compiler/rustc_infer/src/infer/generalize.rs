use rustc_data_structures::sso::SsoHashMap;
use rustc_hir::def_id::DefId;
use rustc_middle::infer::unify_key::{ConstVarValue, ConstVariableValue};
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::relate::{self, Relate, RelateResult, TypeRelation};
use rustc_middle::ty::{self, InferConst, Ty, TyCtxt, TypeVisitableExt};
use rustc_span::Span;

use crate::infer::nll_relate::TypeRelatingDelegate;
use crate::infer::type_variable::TypeVariableValue;
use crate::infer::{InferCtxt, RegionVariableOrigin};

pub(super) fn generalize<'tcx, D: GeneralizerDelegate<'tcx>>(
    infcx: &InferCtxt<'tcx>,
    delegate: &mut D,
    ty: Ty<'tcx>,
    for_vid: ty::TyVid,
    ambient_variance: ty::Variance,
) -> RelateResult<'tcx, Generalization<Ty<'tcx>>> {
    let for_universe = infcx.probe_ty_var(for_vid).unwrap_err();
    let for_vid_sub_root = infcx.inner.borrow_mut().type_variables().sub_root_var(for_vid);

    let mut generalizer = Generalizer {
        infcx,
        delegate,
        ambient_variance,
        for_vid_sub_root,
        for_universe,
        root_ty: ty,
        needs_wf: false,
        cache: Default::default(),
    };

    assert!(!ty.has_escaping_bound_vars());
    let value = generalizer.relate(ty, ty)?;
    let needs_wf = generalizer.needs_wf;
    Ok(Generalization { value, needs_wf })
}

/// Abstracts the handling of region vars between HIR and MIR/NLL typechecking
/// in the generalizer code.
pub trait GeneralizerDelegate<'tcx> {
    fn param_env(&self) -> ty::ParamEnv<'tcx>;

    fn forbid_inference_vars() -> bool;

    fn generalize_existential(&mut self, universe: ty::UniverseIndex) -> ty::Region<'tcx>;
}

pub struct CombineDelegate<'cx, 'tcx> {
    pub infcx: &'cx InferCtxt<'tcx>,
    pub param_env: ty::ParamEnv<'tcx>,
    pub span: Span,
}

impl<'tcx> GeneralizerDelegate<'tcx> for CombineDelegate<'_, 'tcx> {
    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.param_env
    }

    fn forbid_inference_vars() -> bool {
        false
    }

    fn generalize_existential(&mut self, universe: ty::UniverseIndex) -> ty::Region<'tcx> {
        self.infcx
            .next_region_var_in_universe(RegionVariableOrigin::MiscVariable(self.span), universe)
    }
}

impl<'tcx, T> GeneralizerDelegate<'tcx> for T
where
    T: TypeRelatingDelegate<'tcx>,
{
    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        <Self as TypeRelatingDelegate<'tcx>>::param_env(self)
    }

    fn forbid_inference_vars() -> bool {
        <Self as TypeRelatingDelegate<'tcx>>::forbid_inference_vars()
    }

    fn generalize_existential(&mut self, universe: ty::UniverseIndex) -> ty::Region<'tcx> {
        <Self as TypeRelatingDelegate<'tcx>>::generalize_existential(self, universe)
    }
}

/// The "type generalizer" is used when handling inference variables.
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
/// [blog post]: https://is.gd/0hKvIr
struct Generalizer<'me, 'tcx, D>
where
    D: GeneralizerDelegate<'tcx>,
{
    pub infcx: &'me InferCtxt<'tcx>,

    // An delegate used to abstract the behaviors of the three previous
    // generalizer-like implementations.
    pub delegate: &'me mut D,

    /// After we generalize this type, we are going to relate it to
    /// some other type. What will be the variance at this point?
    ambient_variance: ty::Variance,

    /// The vid of the type variable that is in the process of being
    /// instantiated. If we find this within the value we are folding,
    /// that means we would have created a cyclic value.
    pub for_vid_sub_root: ty::TyVid,

    /// The universe of the type variable that is in the process of being
    /// instantiated. If we find anything that this universe cannot name,
    /// we reject the relation.
    for_universe: ty::UniverseIndex,

    pub root_ty: Ty<'tcx>,

    cache: SsoHashMap<Ty<'tcx>, Ty<'tcx>>,

    /// See the field `needs_wf` in `Generalization`.
    needs_wf: bool,
}

impl<'tcx, D> TypeRelation<'tcx> for Generalizer<'_, 'tcx, D>
where
    D: GeneralizerDelegate<'tcx>,
{
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.delegate.param_env()
    }

    fn tag(&self) -> &'static str {
        "Generalizer"
    }

    fn a_is_expected(&self) -> bool {
        true
    }

    fn relate_item_substs(
        &mut self,
        item_def_id: DefId,
        a_subst: ty::SubstsRef<'tcx>,
        b_subst: ty::SubstsRef<'tcx>,
    ) -> RelateResult<'tcx, ty::SubstsRef<'tcx>> {
        if self.ambient_variance == ty::Variance::Invariant {
            // Avoid fetching the variance if we are in an invariant
            // context; no need, and it can induce dependency cycles
            // (e.g., #41849).
            relate::relate_substs(self, a_subst, b_subst)
        } else {
            let tcx = self.tcx();
            let opt_variances = tcx.variances_of(item_def_id);
            relate::relate_substs_with_variances(
                self,
                item_def_id,
                &opt_variances,
                a_subst,
                b_subst,
                true,
            )
        }
    }

    fn relate_with_variance<T: Relate<'tcx>>(
        &mut self,
        variance: ty::Variance,
        _info: ty::VarianceDiagInfo<'tcx>,
        a: T,
        b: T,
    ) -> RelateResult<'tcx, T> {
        debug!("Generalizer::relate_with_variance(variance={:?}, a={:?}, b={:?})", variance, a, b);

        let old_ambient_variance = self.ambient_variance;
        self.ambient_variance = self.ambient_variance.xform(variance);

        debug!("Generalizer::relate_with_variance: ambient_variance = {:?}", self.ambient_variance);

        let r = self.relate(a, b)?;

        self.ambient_variance = old_ambient_variance;

        debug!("Generalizer::relate_with_variance: r={:?}", r);

        Ok(r)
    }

    fn tys(&mut self, t: Ty<'tcx>, t2: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        assert_eq!(t, t2); // we are misusing TypeRelation here; both LHS and RHS ought to be ==

        if let Some(&result) = self.cache.get(&t) {
            return Ok(result);
        }
        debug!("generalize: t={:?}", t);

        // Check to see whether the type we are generalizing references
        // any other type variable related to `vid` via
        // subtyping. This is basically our "occurs check", preventing
        // us from creating infinitely sized types.
        let g = match *t.kind() {
            ty::Infer(ty::TyVar(_)) | ty::Infer(ty::IntVar(_)) | ty::Infer(ty::FloatVar(_))
                if D::forbid_inference_vars() =>
            {
                bug!("unexpected inference variable encountered in NLL generalization: {t}");
            }

            ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
                bug!("unexpected infer type: {t}")
            }

            ty::Infer(ty::TyVar(vid)) => {
                let mut inner = self.infcx.inner.borrow_mut();
                let vid = inner.type_variables().root_var(vid);
                let sub_vid = inner.type_variables().sub_root_var(vid);
                if sub_vid == self.for_vid_sub_root {
                    // If sub-roots are equal, then `for_vid` and
                    // `vid` are related via subtyping.
                    Err(TypeError::CyclicTy(self.root_ty))
                } else {
                    let probe = inner.type_variables().probe(vid);
                    match probe {
                        TypeVariableValue::Known { value: u } => {
                            debug!("generalize: known value {:?}", u);
                            drop(inner);
                            self.relate(u, u)
                        }
                        TypeVariableValue::Unknown { universe } => {
                            match self.ambient_variance {
                                // Invariant: no need to make a fresh type variable.
                                ty::Invariant => {
                                    if self.for_universe.can_name(universe) {
                                        return Ok(t);
                                    }
                                }

                                // Bivariant: make a fresh var, but we
                                // may need a WF predicate. See
                                // comment on `needs_wf` field for
                                // more info.
                                ty::Bivariant => self.needs_wf = true,

                                // Co/contravariant: this will be
                                // sufficiently constrained later on.
                                ty::Covariant | ty::Contravariant => (),
                            }

                            let origin = *inner.type_variables().var_origin(vid);
                            let new_var_id =
                                inner.type_variables().new_var(self.for_universe, origin);
                            let u = self.tcx().mk_ty_var(new_var_id);

                            // Record that we replaced `vid` with `new_var_id` as part of a generalization
                            // operation. This is needed to detect cyclic types. To see why, see the
                            // docs in the `type_variables` module.
                            inner.type_variables().sub(vid, new_var_id);
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
                Ok(t)
            }

            ty::Placeholder(placeholder) => {
                if self.for_universe.cannot_name(placeholder.universe) {
                    debug!(
                        "Generalizer::tys: root universe {:?} cannot name\
                         placeholder in universe {:?}",
                        self.for_universe, placeholder.universe
                    );
                    Err(TypeError::Mismatch)
                } else {
                    Ok(t)
                }
            }

            ty::Alias(ty::Opaque, ty::AliasTy { def_id, substs, .. }) => {
                let s = self.relate(substs, substs)?;
                Ok(if s == substs { t } else { self.tcx().mk_opaque(def_id, s) })
            }
            _ => relate::super_relate_tys(self, t, t),
        }?;

        self.cache.insert(t, g);
        Ok(g)
    }

    fn regions(
        &mut self,
        r: ty::Region<'tcx>,
        r2: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        assert_eq!(r, r2); // we are misusing TypeRelation here; both LHS and RHS ought to be ==

        debug!("generalize: regions r={:?}", r);

        match *r {
            // Never make variables for regions bound within the type itself,
            // nor for erased regions.
            ty::ReLateBound(..) | ty::ReErased => {
                return Ok(r);
            }

            ty::ReError(_) => {
                return Ok(r);
            }

            ty::RePlaceholder(..)
            | ty::ReVar(..)
            | ty::ReStatic
            | ty::ReEarlyBound(..)
            | ty::ReFree(..) => {
                // see common code below
            }
        }

        // If we are in an invariant context, we can re-use the region
        // as is, unless it happens to be in some universe that we
        // can't name.
        if let ty::Invariant = self.ambient_variance {
            let r_universe = self.infcx.universe_of_region(r);
            if self.for_universe.can_name(r_universe) {
                return Ok(r);
            }
        }

        // FIXME: This is non-ideal because we don't give a
        // very descriptive origin for this region variable.
        let replacement_region_vid = self.delegate.generalize_existential(self.for_universe);

        Ok(replacement_region_vid)
    }

    fn consts(
        &mut self,
        c: ty::Const<'tcx>,
        c2: ty::Const<'tcx>,
    ) -> RelateResult<'tcx, ty::Const<'tcx>> {
        assert_eq!(c, c2); // we are misusing TypeRelation here; both LHS and RHS ought to be ==

        match c.kind() {
            ty::ConstKind::Infer(InferConst::Var(_)) if D::forbid_inference_vars() => {
                bug!("unexpected inference variable encountered in NLL generalization: {:?}", c);
            }
            ty::ConstKind::Infer(InferConst::Var(vid)) => {
                let mut inner = self.infcx.inner.borrow_mut();
                let variable_table = &mut inner.const_unification_table();
                let var_value = variable_table.probe_value(vid);
                match var_value.val {
                    ConstVariableValue::Known { value: u } => {
                        drop(inner);
                        self.relate(u, u)
                    }
                    ConstVariableValue::Unknown { universe } => {
                        if self.for_universe.can_name(universe) {
                            Ok(c)
                        } else {
                            let new_var_id = variable_table.new_key(ConstVarValue {
                                origin: var_value.origin,
                                val: ConstVariableValue::Unknown { universe: self.for_universe },
                            });
                            Ok(self.tcx().mk_const(new_var_id, c.ty()))
                        }
                    }
                }
            }
            // FIXME: remove this branch once `structurally_relate_consts` is fully
            // structural.
            ty::ConstKind::Unevaluated(ty::UnevaluatedConst { def, substs }) => {
                let substs = self.relate_with_variance(
                    ty::Variance::Invariant,
                    ty::VarianceDiagInfo::default(),
                    substs,
                    substs,
                )?;
                Ok(self.tcx().mk_const(ty::UnevaluatedConst { def, substs }, c.ty()))
            }
            _ => relate::super_relate_consts(self, c, c),
        }
    }

    fn binders<T>(
        &mut self,
        a: ty::Binder<'tcx, T>,
        _: ty::Binder<'tcx, T>,
    ) -> RelateResult<'tcx, ty::Binder<'tcx, T>>
    where
        T: Relate<'tcx>,
    {
        debug!("Generalizer::binders(a={:?})", a);
        let result = self.relate(a.skip_binder(), a.skip_binder())?;
        Ok(a.rebind(result))
    }
}

/// Result from a generalization operation. This includes
/// not only the generalized type, but also a bool flag
/// indicating whether further WF checks are needed.
#[derive(Debug)]
pub struct Generalization<T> {
    pub value: T,

    /// If true, then the generalized type may not be well-formed,
    /// even if the source type is well-formed, so we should add an
    /// additional check to enforce that it is. This arises in
    /// particular around 'bivariant' type parameters that are only
    /// constrained by a where-clause. As an example, imagine a type:
    ///
    ///     struct Foo<A, B> where A: Iterator<Item = B> {
    ///         data: A
    ///     }
    ///
    /// here, `A` will be covariant, but `B` is
    /// unconstrained. However, whatever it is, for `Foo` to be WF, it
    /// must be equal to `A::Item`. If we have an input `Foo<?A, ?B>`,
    /// then after generalization we will wind up with a type like
    /// `Foo<?C, ?D>`. When we enforce that `Foo<?A, ?B> <: Foo<?C,
    /// ?D>` (or `>:`), we will wind up with the requirement that `?A
    /// <: ?C`, but no particular relationship between `?B` and `?D`
    /// (after all, we do not know the variance of the normalized form
    /// of `A::Item` with respect to `A`). If we do nothing else, this
    /// may mean that `?D` goes unconstrained (as in #41677). So, in
    /// this scenario where we create a new type variable in a
    /// bivariant context, we set the `needs_wf` flag to true. This
    /// will force the calling code to check that `WF(Foo<?C, ?D>)`
    /// holds, which in turn implies that `?C::Item == ?D`. So once
    /// `?C` is constrained, that should suffice to restrict `?D`.
    pub needs_wf: bool,
}
