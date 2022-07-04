//! Generalized type relating mechanism.
//!
//! A type relation `R` relates a pair of values `(A, B)`. `A and B` are usually
//! types or regions but can be other things. Examples of type relations are
//! subtyping, type equality, etc.

use crate::ty::error::{ExpectedFound, TypeError};
use crate::ty::subst::{GenericArg, GenericArgKind, Subst, SubstsRef};
use crate::ty::{self, ImplSubject, Term, TermKind, Ty, TyCtxt, TypeFoldable};
use rustc_hir as ast;
use rustc_hir::def_id::DefId;
use rustc_span::DUMMY_SP;
use rustc_target::spec::abi;
use std::iter;

pub type RelateResult<'tcx, T> = Result<T, TypeError<'tcx>>;

#[derive(Clone, Debug)]
pub enum Cause {
    ExistentialRegionBound, // relating an existential region bound
}

pub trait TypeRelation<'tcx>: Sized {
    fn tcx(&self) -> TyCtxt<'tcx>;

    fn param_env(&self) -> ty::ParamEnv<'tcx>;

    /// Returns a static string we can use for printouts.
    fn tag(&self) -> &'static str;

    /// Returns `true` if the value `a` is the "expected" type in the
    /// relation. Just affects error messages.
    fn a_is_expected(&self) -> bool;

    fn with_cause<F, R>(&mut self, _cause: Cause, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        f(self)
    }

    /// Generic relation routine suitable for most anything.
    fn relate<T: Relate<'tcx>>(&mut self, a: T, b: T) -> RelateResult<'tcx, T> {
        Relate::relate(self, a, b)
    }

    /// Relate the two substitutions for the given item. The default
    /// is to look up the variance for the item and proceed
    /// accordingly.
    fn relate_item_substs(
        &mut self,
        item_def_id: DefId,
        a_subst: SubstsRef<'tcx>,
        b_subst: SubstsRef<'tcx>,
    ) -> RelateResult<'tcx, SubstsRef<'tcx>> {
        debug!(
            "relate_item_substs(item_def_id={:?}, a_subst={:?}, b_subst={:?})",
            item_def_id, a_subst, b_subst
        );

        let tcx = self.tcx();
        let opt_variances = tcx.variances_of(item_def_id);
        relate_substs_with_variances(self, item_def_id, opt_variances, a_subst, b_subst)
    }

    /// Switch variance for the purpose of relating `a` and `b`.
    fn relate_with_variance<T: Relate<'tcx>>(
        &mut self,
        variance: ty::Variance,
        info: ty::VarianceDiagInfo<'tcx>,
        a: T,
        b: T,
    ) -> RelateResult<'tcx, T>;

    // Overridable relations. You shouldn't typically call these
    // directly, instead call `relate()`, which in turn calls
    // these. This is both more uniform but also allows us to add
    // additional hooks for other types in the future if needed
    // without making older code, which called `relate`, obsolete.

    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>>;

    fn regions(
        &mut self,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>>;

    fn consts(
        &mut self,
        a: ty::Const<'tcx>,
        b: ty::Const<'tcx>,
    ) -> RelateResult<'tcx, ty::Const<'tcx>>;

    fn binders<T>(
        &mut self,
        a: ty::Binder<'tcx, T>,
        b: ty::Binder<'tcx, T>,
    ) -> RelateResult<'tcx, ty::Binder<'tcx, T>>
    where
        T: Relate<'tcx>;
}

pub trait Relate<'tcx>: TypeFoldable<'tcx> + Copy {
    fn relate<R: TypeRelation<'tcx>>(
        relation: &mut R,
        a: Self,
        b: Self,
    ) -> RelateResult<'tcx, Self>;
}

///////////////////////////////////////////////////////////////////////////
// Relate impls

pub fn relate_type_and_mut<'tcx, R: TypeRelation<'tcx>>(
    relation: &mut R,
    a: ty::TypeAndMut<'tcx>,
    b: ty::TypeAndMut<'tcx>,
    base_ty: Ty<'tcx>,
) -> RelateResult<'tcx, ty::TypeAndMut<'tcx>> {
    debug!("{}.mts({:?}, {:?})", relation.tag(), a, b);
    if a.mutbl != b.mutbl {
        Err(TypeError::Mutability)
    } else {
        let mutbl = a.mutbl;
        let (variance, info) = match mutbl {
            ast::Mutability::Not => (ty::Covariant, ty::VarianceDiagInfo::None),
            ast::Mutability::Mut => {
                (ty::Invariant, ty::VarianceDiagInfo::Invariant { ty: base_ty, param_index: 0 })
            }
        };
        let ty = relation.relate_with_variance(variance, info, a.ty, b.ty)?;
        Ok(ty::TypeAndMut { ty, mutbl })
    }
}

#[inline]
pub fn relate_substs<'tcx, R: TypeRelation<'tcx>>(
    relation: &mut R,
    a_subst: SubstsRef<'tcx>,
    b_subst: SubstsRef<'tcx>,
) -> RelateResult<'tcx, SubstsRef<'tcx>> {
    relation.tcx().mk_substs(iter::zip(a_subst, b_subst).map(|(a, b)| {
        relation.relate_with_variance(ty::Invariant, ty::VarianceDiagInfo::default(), a, b)
    }))
}

pub fn relate_substs_with_variances<'tcx, R: TypeRelation<'tcx>>(
    relation: &mut R,
    ty_def_id: DefId,
    variances: &[ty::Variance],
    a_subst: SubstsRef<'tcx>,
    b_subst: SubstsRef<'tcx>,
) -> RelateResult<'tcx, SubstsRef<'tcx>> {
    let tcx = relation.tcx();

    let mut cached_ty = None;
    let params = iter::zip(a_subst, b_subst).enumerate().map(|(i, (a, b))| {
        let variance = variances[i];
        let variance_info = if variance == ty::Invariant {
            let ty =
                *cached_ty.get_or_insert_with(|| tcx.bound_type_of(ty_def_id).subst(tcx, a_subst));
            ty::VarianceDiagInfo::Invariant { ty, param_index: i.try_into().unwrap() }
        } else {
            ty::VarianceDiagInfo::default()
        };
        relation.relate_with_variance(variance, variance_info, a, b)
    });

    tcx.mk_substs(params)
}

impl<'tcx> Relate<'tcx> for ty::FnSig<'tcx> {
    fn relate<R: TypeRelation<'tcx>>(
        relation: &mut R,
        a: ty::FnSig<'tcx>,
        b: ty::FnSig<'tcx>,
    ) -> RelateResult<'tcx, ty::FnSig<'tcx>> {
        let tcx = relation.tcx();

        if a.c_variadic != b.c_variadic {
            return Err(TypeError::VariadicMismatch(expected_found(
                relation,
                a.c_variadic,
                b.c_variadic,
            )));
        }
        let unsafety = relation.relate(a.unsafety, b.unsafety)?;
        let abi = relation.relate(a.abi, b.abi)?;

        if a.inputs().len() != b.inputs().len() {
            return Err(TypeError::ArgCount);
        }

        let inputs_and_output = iter::zip(a.inputs(), b.inputs())
            .map(|(&a, &b)| ((a, b), false))
            .chain(iter::once(((a.output(), b.output()), true)))
            .map(|((a, b), is_output)| {
                if is_output {
                    relation.relate(a, b)
                } else {
                    relation.relate_with_variance(
                        ty::Contravariant,
                        ty::VarianceDiagInfo::default(),
                        a,
                        b,
                    )
                }
            })
            .enumerate()
            .map(|(i, r)| match r {
                Err(TypeError::Sorts(exp_found) | TypeError::ArgumentSorts(exp_found, _)) => {
                    Err(TypeError::ArgumentSorts(exp_found, i))
                }
                Err(TypeError::Mutability | TypeError::ArgumentMutability(_)) => {
                    Err(TypeError::ArgumentMutability(i))
                }
                r => r,
            });
        Ok(ty::FnSig {
            inputs_and_output: tcx.mk_type_list(inputs_and_output)?,
            c_variadic: a.c_variadic,
            unsafety,
            abi,
        })
    }
}

impl<'tcx> Relate<'tcx> for ty::BoundConstness {
    fn relate<R: TypeRelation<'tcx>>(
        relation: &mut R,
        a: ty::BoundConstness,
        b: ty::BoundConstness,
    ) -> RelateResult<'tcx, ty::BoundConstness> {
        if a != b {
            Err(TypeError::ConstnessMismatch(expected_found(relation, a, b)))
        } else {
            Ok(a)
        }
    }
}

impl<'tcx> Relate<'tcx> for ast::Unsafety {
    fn relate<R: TypeRelation<'tcx>>(
        relation: &mut R,
        a: ast::Unsafety,
        b: ast::Unsafety,
    ) -> RelateResult<'tcx, ast::Unsafety> {
        if a != b {
            Err(TypeError::UnsafetyMismatch(expected_found(relation, a, b)))
        } else {
            Ok(a)
        }
    }
}

impl<'tcx> Relate<'tcx> for abi::Abi {
    fn relate<R: TypeRelation<'tcx>>(
        relation: &mut R,
        a: abi::Abi,
        b: abi::Abi,
    ) -> RelateResult<'tcx, abi::Abi> {
        if a == b { Ok(a) } else { Err(TypeError::AbiMismatch(expected_found(relation, a, b))) }
    }
}

impl<'tcx> Relate<'tcx> for ty::ProjectionTy<'tcx> {
    fn relate<R: TypeRelation<'tcx>>(
        relation: &mut R,
        a: ty::ProjectionTy<'tcx>,
        b: ty::ProjectionTy<'tcx>,
    ) -> RelateResult<'tcx, ty::ProjectionTy<'tcx>> {
        if a.item_def_id != b.item_def_id {
            Err(TypeError::ProjectionMismatched(expected_found(
                relation,
                a.item_def_id,
                b.item_def_id,
            )))
        } else {
            let substs = relation.relate(a.substs, b.substs)?;
            Ok(ty::ProjectionTy { item_def_id: a.item_def_id, substs: &substs })
        }
    }
}

impl<'tcx> Relate<'tcx> for ty::ExistentialProjection<'tcx> {
    fn relate<R: TypeRelation<'tcx>>(
        relation: &mut R,
        a: ty::ExistentialProjection<'tcx>,
        b: ty::ExistentialProjection<'tcx>,
    ) -> RelateResult<'tcx, ty::ExistentialProjection<'tcx>> {
        if a.item_def_id != b.item_def_id {
            Err(TypeError::ProjectionMismatched(expected_found(
                relation,
                a.item_def_id,
                b.item_def_id,
            )))
        } else {
            let term = relation.relate_with_variance(
                ty::Invariant,
                ty::VarianceDiagInfo::default(),
                a.term,
                b.term,
            )?;
            let substs = relation.relate_with_variance(
                ty::Invariant,
                ty::VarianceDiagInfo::default(),
                a.substs,
                b.substs,
            )?;
            Ok(ty::ExistentialProjection { item_def_id: a.item_def_id, substs, term })
        }
    }
}

impl<'tcx> Relate<'tcx> for ty::TraitRef<'tcx> {
    fn relate<R: TypeRelation<'tcx>>(
        relation: &mut R,
        a: ty::TraitRef<'tcx>,
        b: ty::TraitRef<'tcx>,
    ) -> RelateResult<'tcx, ty::TraitRef<'tcx>> {
        // Different traits cannot be related.
        if a.def_id != b.def_id {
            Err(TypeError::Traits(expected_found(relation, a.def_id, b.def_id)))
        } else {
            let substs = relate_substs(relation, a.substs, b.substs)?;
            Ok(ty::TraitRef { def_id: a.def_id, substs })
        }
    }
}

impl<'tcx> Relate<'tcx> for ty::ExistentialTraitRef<'tcx> {
    fn relate<R: TypeRelation<'tcx>>(
        relation: &mut R,
        a: ty::ExistentialTraitRef<'tcx>,
        b: ty::ExistentialTraitRef<'tcx>,
    ) -> RelateResult<'tcx, ty::ExistentialTraitRef<'tcx>> {
        // Different traits cannot be related.
        if a.def_id != b.def_id {
            Err(TypeError::Traits(expected_found(relation, a.def_id, b.def_id)))
        } else {
            let substs = relate_substs(relation, a.substs, b.substs)?;
            Ok(ty::ExistentialTraitRef { def_id: a.def_id, substs })
        }
    }
}

#[derive(Copy, Debug, Clone, TypeFoldable, TypeVisitable)]
struct GeneratorWitness<'tcx>(&'tcx ty::List<Ty<'tcx>>);

impl<'tcx> Relate<'tcx> for GeneratorWitness<'tcx> {
    fn relate<R: TypeRelation<'tcx>>(
        relation: &mut R,
        a: GeneratorWitness<'tcx>,
        b: GeneratorWitness<'tcx>,
    ) -> RelateResult<'tcx, GeneratorWitness<'tcx>> {
        assert_eq!(a.0.len(), b.0.len());
        let tcx = relation.tcx();
        let types = tcx.mk_type_list(iter::zip(a.0, b.0).map(|(a, b)| relation.relate(a, b)))?;
        Ok(GeneratorWitness(types))
    }
}

impl<'tcx> Relate<'tcx> for ImplSubject<'tcx> {
    #[inline]
    fn relate<R: TypeRelation<'tcx>>(
        relation: &mut R,
        a: ImplSubject<'tcx>,
        b: ImplSubject<'tcx>,
    ) -> RelateResult<'tcx, ImplSubject<'tcx>> {
        match (a, b) {
            (ImplSubject::Trait(trait_ref_a), ImplSubject::Trait(trait_ref_b)) => {
                let trait_ref = ty::TraitRef::relate(relation, trait_ref_a, trait_ref_b)?;
                Ok(ImplSubject::Trait(trait_ref))
            }
            (ImplSubject::Inherent(ty_a), ImplSubject::Inherent(ty_b)) => {
                let ty = Ty::relate(relation, ty_a, ty_b)?;
                Ok(ImplSubject::Inherent(ty))
            }
            (ImplSubject::Trait(_), ImplSubject::Inherent(_))
            | (ImplSubject::Inherent(_), ImplSubject::Trait(_)) => {
                bug!("can not relate TraitRef and Ty");
            }
        }
    }
}

impl<'tcx> Relate<'tcx> for Ty<'tcx> {
    #[inline]
    fn relate<R: TypeRelation<'tcx>>(
        relation: &mut R,
        a: Ty<'tcx>,
        b: Ty<'tcx>,
    ) -> RelateResult<'tcx, Ty<'tcx>> {
        relation.tys(a, b)
    }
}

/// The main "type relation" routine. Note that this does not handle
/// inference artifacts, so you should filter those out before calling
/// it.
pub fn super_relate_tys<'tcx, R: TypeRelation<'tcx>>(
    relation: &mut R,
    a: Ty<'tcx>,
    b: Ty<'tcx>,
) -> RelateResult<'tcx, Ty<'tcx>> {
    let tcx = relation.tcx();
    debug!("super_relate_tys: a={:?} b={:?}", a, b);
    match (a.kind(), b.kind()) {
        (&ty::Infer(_), _) | (_, &ty::Infer(_)) => {
            // The caller should handle these cases!
            bug!("var types encountered in super_relate_tys")
        }

        (ty::Bound(..), _) | (_, ty::Bound(..)) => {
            bug!("bound types encountered in super_relate_tys")
        }

        (&ty::Error(_), _) | (_, &ty::Error(_)) => Ok(tcx.ty_error()),

        (&ty::Never, _)
        | (&ty::Char, _)
        | (&ty::Bool, _)
        | (&ty::Int(_), _)
        | (&ty::Uint(_), _)
        | (&ty::Float(_), _)
        | (&ty::Str, _)
            if a == b =>
        {
            Ok(a)
        }

        (&ty::Param(ref a_p), &ty::Param(ref b_p)) if a_p.index == b_p.index => Ok(a),

        (ty::Placeholder(p1), ty::Placeholder(p2)) if p1 == p2 => Ok(a),

        (&ty::Adt(a_def, a_substs), &ty::Adt(b_def, b_substs)) if a_def == b_def => {
            let substs = relation.relate_item_substs(a_def.did(), a_substs, b_substs)?;
            Ok(tcx.mk_adt(a_def, substs))
        }

        (&ty::Foreign(a_id), &ty::Foreign(b_id)) if a_id == b_id => Ok(tcx.mk_foreign(a_id)),

        (&ty::Dynamic(a_obj, a_region), &ty::Dynamic(b_obj, b_region)) => {
            let region_bound = relation.with_cause(Cause::ExistentialRegionBound, |relation| {
                relation.relate_with_variance(
                    ty::Contravariant,
                    ty::VarianceDiagInfo::default(),
                    a_region,
                    b_region,
                )
            })?;
            Ok(tcx.mk_dynamic(relation.relate(a_obj, b_obj)?, region_bound))
        }

        (&ty::Generator(a_id, a_substs, movability), &ty::Generator(b_id, b_substs, _))
            if a_id == b_id =>
        {
            // All Generator types with the same id represent
            // the (anonymous) type of the same generator expression. So
            // all of their regions should be equated.
            let substs = relation.relate(a_substs, b_substs)?;
            Ok(tcx.mk_generator(a_id, substs, movability))
        }

        (&ty::GeneratorWitness(a_types), &ty::GeneratorWitness(b_types)) => {
            // Wrap our types with a temporary GeneratorWitness struct
            // inside the binder so we can related them
            let a_types = a_types.map_bound(GeneratorWitness);
            let b_types = b_types.map_bound(GeneratorWitness);
            // Then remove the GeneratorWitness for the result
            let types = relation.relate(a_types, b_types)?.map_bound(|witness| witness.0);
            Ok(tcx.mk_generator_witness(types))
        }

        (&ty::Closure(a_id, a_substs), &ty::Closure(b_id, b_substs)) if a_id == b_id => {
            // All Closure types with the same id represent
            // the (anonymous) type of the same closure expression. So
            // all of their regions should be equated.
            let substs = relation.relate(a_substs, b_substs)?;
            Ok(tcx.mk_closure(a_id, &substs))
        }

        (&ty::RawPtr(a_mt), &ty::RawPtr(b_mt)) => {
            let mt = relate_type_and_mut(relation, a_mt, b_mt, a)?;
            Ok(tcx.mk_ptr(mt))
        }

        (&ty::Ref(a_r, a_ty, a_mutbl), &ty::Ref(b_r, b_ty, b_mutbl)) => {
            let r = relation.relate_with_variance(
                ty::Contravariant,
                ty::VarianceDiagInfo::default(),
                a_r,
                b_r,
            )?;
            let a_mt = ty::TypeAndMut { ty: a_ty, mutbl: a_mutbl };
            let b_mt = ty::TypeAndMut { ty: b_ty, mutbl: b_mutbl };
            let mt = relate_type_and_mut(relation, a_mt, b_mt, a)?;
            Ok(tcx.mk_ref(r, mt))
        }

        (&ty::Array(a_t, sz_a), &ty::Array(b_t, sz_b)) => {
            let t = relation.relate(a_t, b_t)?;
            match relation.relate(sz_a, sz_b) {
                Ok(sz) => Ok(tcx.mk_ty(ty::Array(t, sz))),
                Err(err) => {
                    // Check whether the lengths are both concrete/known values,
                    // but are unequal, for better diagnostics.
                    //
                    // It might seem dubious to eagerly evaluate these constants here,
                    // we however cannot end up with errors in `Relate` during both
                    // `type_of` and `predicates_of`. This means that evaluating the
                    // constants should not cause cycle errors here.
                    let sz_a = sz_a.try_eval_usize(tcx, relation.param_env());
                    let sz_b = sz_b.try_eval_usize(tcx, relation.param_env());
                    match (sz_a, sz_b) {
                        (Some(sz_a_val), Some(sz_b_val)) if sz_a_val != sz_b_val => Err(
                            TypeError::FixedArraySize(expected_found(relation, sz_a_val, sz_b_val)),
                        ),
                        _ => Err(err),
                    }
                }
            }
        }

        (&ty::Slice(a_t), &ty::Slice(b_t)) => {
            let t = relation.relate(a_t, b_t)?;
            Ok(tcx.mk_slice(t))
        }

        (&ty::Tuple(as_), &ty::Tuple(bs)) => {
            if as_.len() == bs.len() {
                Ok(tcx.mk_tup(iter::zip(as_, bs).map(|(a, b)| relation.relate(a, b)))?)
            } else if !(as_.is_empty() || bs.is_empty()) {
                Err(TypeError::TupleSize(expected_found(relation, as_.len(), bs.len())))
            } else {
                Err(TypeError::Sorts(expected_found(relation, a, b)))
            }
        }

        (&ty::FnDef(a_def_id, a_substs), &ty::FnDef(b_def_id, b_substs))
            if a_def_id == b_def_id =>
        {
            let substs = relation.relate_item_substs(a_def_id, a_substs, b_substs)?;
            Ok(tcx.mk_fn_def(a_def_id, substs))
        }

        (&ty::FnPtr(a_fty), &ty::FnPtr(b_fty)) => {
            let fty = relation.relate(a_fty, b_fty)?;
            Ok(tcx.mk_fn_ptr(fty))
        }

        // these two are already handled downstream in case of lazy normalization
        (&ty::Projection(a_data), &ty::Projection(b_data)) => {
            let projection_ty = relation.relate(a_data, b_data)?;
            Ok(tcx.mk_projection(projection_ty.item_def_id, projection_ty.substs))
        }

        (&ty::Opaque(a_def_id, a_substs), &ty::Opaque(b_def_id, b_substs))
            if a_def_id == b_def_id =>
        {
            let substs = relate_substs(relation, a_substs, b_substs)?;
            Ok(tcx.mk_opaque(a_def_id, substs))
        }

        _ => Err(TypeError::Sorts(expected_found(relation, a, b))),
    }
}

/// The main "const relation" routine. Note that this does not handle
/// inference artifacts, so you should filter those out before calling
/// it.
pub fn super_relate_consts<'tcx, R: TypeRelation<'tcx>>(
    relation: &mut R,
    a: ty::Const<'tcx>,
    b: ty::Const<'tcx>,
) -> RelateResult<'tcx, ty::Const<'tcx>> {
    debug!("{}.super_relate_consts(a = {:?}, b = {:?})", relation.tag(), a, b);
    let tcx = relation.tcx();

    let a_ty;
    let b_ty;
    if relation.tcx().features().adt_const_params {
        a_ty = tcx.normalize_erasing_regions(relation.param_env(), a.ty());
        b_ty = tcx.normalize_erasing_regions(relation.param_env(), b.ty());
    } else {
        a_ty = tcx.erase_regions(a.ty());
        b_ty = tcx.erase_regions(b.ty());
    }
    if a_ty != b_ty {
        relation.tcx().sess.delay_span_bug(
            DUMMY_SP,
            &format!("cannot relate constants of different types: {} != {}", a_ty, b_ty),
        );
    }

    // Currently, the values that can be unified are primitive types,
    // and those that derive both `PartialEq` and `Eq`, corresponding
    // to structural-match types.
    let is_match = match (a.kind(), b.kind()) {
        (ty::ConstKind::Infer(_), _) | (_, ty::ConstKind::Infer(_)) => {
            // The caller should handle these cases!
            bug!("var types encountered in super_relate_consts: {:?} {:?}", a, b)
        }

        (ty::ConstKind::Error(_), _) => return Ok(a),
        (_, ty::ConstKind::Error(_)) => return Ok(b),

        (ty::ConstKind::Param(a_p), ty::ConstKind::Param(b_p)) => a_p.index == b_p.index,
        (ty::ConstKind::Placeholder(p1), ty::ConstKind::Placeholder(p2)) => p1 == p2,
        (ty::ConstKind::Value(a_val), ty::ConstKind::Value(b_val)) => a_val == b_val,

        (ty::ConstKind::Unevaluated(au), ty::ConstKind::Unevaluated(bu))
            if tcx.features().generic_const_exprs =>
        {
            tcx.try_unify_abstract_consts(relation.param_env().and((au.shrink(), bu.shrink())))
        }

        // While this is slightly incorrect, it shouldn't matter for `min_const_generics`
        // and is the better alternative to waiting until `generic_const_exprs` can
        // be stabilized.
        (ty::ConstKind::Unevaluated(au), ty::ConstKind::Unevaluated(bu))
            if au.def == bu.def && au.promoted == bu.promoted =>
        {
            let substs = relation.relate_with_variance(
                ty::Variance::Invariant,
                ty::VarianceDiagInfo::default(),
                au.substs,
                bu.substs,
            )?;
            return Ok(tcx.mk_const(ty::ConstS {
                kind: ty::ConstKind::Unevaluated(ty::Unevaluated {
                    def: au.def,
                    substs,
                    promoted: au.promoted,
                }),
                ty: a.ty(),
            }));
        }
        _ => false,
    };
    if is_match { Ok(a) } else { Err(TypeError::ConstMismatch(expected_found(relation, a, b))) }
}

impl<'tcx> Relate<'tcx> for &'tcx ty::List<ty::Binder<'tcx, ty::ExistentialPredicate<'tcx>>> {
    fn relate<R: TypeRelation<'tcx>>(
        relation: &mut R,
        a: Self,
        b: Self,
    ) -> RelateResult<'tcx, Self> {
        let tcx = relation.tcx();

        // FIXME: this is wasteful, but want to do a perf run to see how slow it is.
        // We need to perform this deduplication as we sometimes generate duplicate projections
        // in `a`.
        let mut a_v: Vec<_> = a.into_iter().collect();
        let mut b_v: Vec<_> = b.into_iter().collect();
        // `skip_binder` here is okay because `stable_cmp` doesn't look at binders
        a_v.sort_by(|a, b| a.skip_binder().stable_cmp(tcx, &b.skip_binder()));
        a_v.dedup();
        b_v.sort_by(|a, b| a.skip_binder().stable_cmp(tcx, &b.skip_binder()));
        b_v.dedup();
        if a_v.len() != b_v.len() {
            return Err(TypeError::ExistentialMismatch(expected_found(relation, a, b)));
        }

        let v = iter::zip(a_v, b_v).map(|(ep_a, ep_b)| {
            use crate::ty::ExistentialPredicate::*;
            match (ep_a.skip_binder(), ep_b.skip_binder()) {
                (Trait(a), Trait(b)) => Ok(ep_a
                    .rebind(Trait(relation.relate(ep_a.rebind(a), ep_b.rebind(b))?.skip_binder()))),
                (Projection(a), Projection(b)) => Ok(ep_a.rebind(Projection(
                    relation.relate(ep_a.rebind(a), ep_b.rebind(b))?.skip_binder(),
                ))),
                (AutoTrait(a), AutoTrait(b)) if a == b => Ok(ep_a.rebind(AutoTrait(a))),
                _ => Err(TypeError::ExistentialMismatch(expected_found(relation, a, b))),
            }
        });
        tcx.mk_poly_existential_predicates(v)
    }
}

impl<'tcx> Relate<'tcx> for ty::ClosureSubsts<'tcx> {
    fn relate<R: TypeRelation<'tcx>>(
        relation: &mut R,
        a: ty::ClosureSubsts<'tcx>,
        b: ty::ClosureSubsts<'tcx>,
    ) -> RelateResult<'tcx, ty::ClosureSubsts<'tcx>> {
        let substs = relate_substs(relation, a.substs, b.substs)?;
        Ok(ty::ClosureSubsts { substs })
    }
}

impl<'tcx> Relate<'tcx> for ty::GeneratorSubsts<'tcx> {
    fn relate<R: TypeRelation<'tcx>>(
        relation: &mut R,
        a: ty::GeneratorSubsts<'tcx>,
        b: ty::GeneratorSubsts<'tcx>,
    ) -> RelateResult<'tcx, ty::GeneratorSubsts<'tcx>> {
        let substs = relate_substs(relation, a.substs, b.substs)?;
        Ok(ty::GeneratorSubsts { substs })
    }
}

impl<'tcx> Relate<'tcx> for SubstsRef<'tcx> {
    fn relate<R: TypeRelation<'tcx>>(
        relation: &mut R,
        a: SubstsRef<'tcx>,
        b: SubstsRef<'tcx>,
    ) -> RelateResult<'tcx, SubstsRef<'tcx>> {
        relate_substs(relation, a, b)
    }
}

impl<'tcx> Relate<'tcx> for ty::Region<'tcx> {
    fn relate<R: TypeRelation<'tcx>>(
        relation: &mut R,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        relation.regions(a, b)
    }
}

impl<'tcx> Relate<'tcx> for ty::Const<'tcx> {
    fn relate<R: TypeRelation<'tcx>>(
        relation: &mut R,
        a: ty::Const<'tcx>,
        b: ty::Const<'tcx>,
    ) -> RelateResult<'tcx, ty::Const<'tcx>> {
        relation.consts(a, b)
    }
}

impl<'tcx, T: Relate<'tcx>> Relate<'tcx> for ty::Binder<'tcx, T> {
    fn relate<R: TypeRelation<'tcx>>(
        relation: &mut R,
        a: ty::Binder<'tcx, T>,
        b: ty::Binder<'tcx, T>,
    ) -> RelateResult<'tcx, ty::Binder<'tcx, T>> {
        relation.binders(a, b)
    }
}

impl<'tcx> Relate<'tcx> for GenericArg<'tcx> {
    fn relate<R: TypeRelation<'tcx>>(
        relation: &mut R,
        a: GenericArg<'tcx>,
        b: GenericArg<'tcx>,
    ) -> RelateResult<'tcx, GenericArg<'tcx>> {
        match (a.unpack(), b.unpack()) {
            (GenericArgKind::Lifetime(a_lt), GenericArgKind::Lifetime(b_lt)) => {
                Ok(relation.relate(a_lt, b_lt)?.into())
            }
            (GenericArgKind::Type(a_ty), GenericArgKind::Type(b_ty)) => {
                Ok(relation.relate(a_ty, b_ty)?.into())
            }
            (GenericArgKind::Const(a_ct), GenericArgKind::Const(b_ct)) => {
                Ok(relation.relate(a_ct, b_ct)?.into())
            }
            (GenericArgKind::Lifetime(unpacked), x) => {
                bug!("impossible case reached: can't relate: {:?} with {:?}", unpacked, x)
            }
            (GenericArgKind::Type(unpacked), x) => {
                bug!("impossible case reached: can't relate: {:?} with {:?}", unpacked, x)
            }
            (GenericArgKind::Const(unpacked), x) => {
                bug!("impossible case reached: can't relate: {:?} with {:?}", unpacked, x)
            }
        }
    }
}

impl<'tcx> Relate<'tcx> for ty::ImplPolarity {
    fn relate<R: TypeRelation<'tcx>>(
        relation: &mut R,
        a: ty::ImplPolarity,
        b: ty::ImplPolarity,
    ) -> RelateResult<'tcx, ty::ImplPolarity> {
        if a != b {
            Err(TypeError::PolarityMismatch(expected_found(relation, a, b)))
        } else {
            Ok(a)
        }
    }
}

impl<'tcx> Relate<'tcx> for ty::TraitPredicate<'tcx> {
    fn relate<R: TypeRelation<'tcx>>(
        relation: &mut R,
        a: ty::TraitPredicate<'tcx>,
        b: ty::TraitPredicate<'tcx>,
    ) -> RelateResult<'tcx, ty::TraitPredicate<'tcx>> {
        Ok(ty::TraitPredicate {
            trait_ref: relation.relate(a.trait_ref, b.trait_ref)?,
            constness: relation.relate(a.constness, b.constness)?,
            polarity: relation.relate(a.polarity, b.polarity)?,
        })
    }
}

impl<'tcx> Relate<'tcx> for Term<'tcx> {
    fn relate<R: TypeRelation<'tcx>>(
        relation: &mut R,
        a: Self,
        b: Self,
    ) -> RelateResult<'tcx, Self> {
        Ok(match (a.unpack(), b.unpack()) {
            (TermKind::Ty(a), TermKind::Ty(b)) => relation.relate(a, b)?.into(),
            (TermKind::Const(a), TermKind::Const(b)) => relation.relate(a, b)?.into(),
            _ => return Err(TypeError::Mismatch),
        })
    }
}

impl<'tcx> Relate<'tcx> for ty::ProjectionPredicate<'tcx> {
    fn relate<R: TypeRelation<'tcx>>(
        relation: &mut R,
        a: ty::ProjectionPredicate<'tcx>,
        b: ty::ProjectionPredicate<'tcx>,
    ) -> RelateResult<'tcx, ty::ProjectionPredicate<'tcx>> {
        Ok(ty::ProjectionPredicate {
            projection_ty: relation.relate(a.projection_ty, b.projection_ty)?,
            term: relation.relate(a.term, b.term)?,
        })
    }
}

///////////////////////////////////////////////////////////////////////////
// Error handling

pub fn expected_found<'tcx, R, T>(relation: &mut R, a: T, b: T) -> ExpectedFound<T>
where
    R: TypeRelation<'tcx>,
{
    ExpectedFound::new(relation.a_is_expected(), a, b)
}
