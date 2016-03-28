// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Generalized type relating mechanism. A type relation R relates a
//! pair of values (A, B). A and B are usually types or regions but
//! can be other things. Examples of type relations are subtyping,
//! type equality, etc.

use middle::def_id::DefId;
use ty::subst::{ParamSpace, Substs};
use ty::{self, Ty, TyCtxt, TypeFoldable};
use ty::error::{ExpectedFound, TypeError};
use std::rc::Rc;
use syntax::abi;
use rustc_front::hir as ast;

pub type RelateResult<'tcx, T> = Result<T, TypeError<'tcx>>;

#[derive(Clone, Debug)]
pub enum Cause {
    ExistentialRegionBound, // relating an existential region bound
}

/// S is the type of extra side effects that may be collected during type relation.
pub trait TypeRelation<'a, 'tcx, S>: Sized {
    fn tcx(&self) -> &'a TyCtxt<'tcx>;

    /// Returns a static string we can use for printouts.
    fn tag(&self) -> &'static str;

    /// Returns true if the value `a` is the "expected" type in the
    /// relation. Just affects error messages.
    fn a_is_expected(&self) -> bool;

    fn with_cause<F,R>(&mut self, _cause: Cause, f: F) -> R
        where F: FnOnce(&mut Self) -> R
    {
        f(self)
    }

    /// Generic relation routine suitable for most anything.
    fn relate<T: Relate<'a, 'tcx>>(&mut self, a: &T, b: &T,
                                   side_effects: &mut S)
        -> RelateResult<'tcx, T>
    {
        Relate::relate(self, a, b, side_effects)
    }

    /// Relete elements of two slices pairwise.
    fn relate_zip<T: Relate<'a, 'tcx>>(&mut self, a: &[T], b: &[T],
                                       side_effects: &mut S)
        -> RelateResult<'tcx, Vec<T>>
    {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(a, b)| self.relate(a, b, side_effects)).collect()
    }

    /// Switch variance for the purpose of relating `a` and `b`.
    fn relate_with_variance<T: Relate<'a, 'tcx>>(&mut self,
                                                 variance: ty::Variance,
                                                 a: &T,
                                                 b: &T,
                                                 side_effects: &mut S)
                                                 -> RelateResult<'tcx, T>;

    // Overrideable relations. You shouldn't typically call these
    // directly, instead call `relate()`, which in turn calls
    // these. This is both more uniform but also allows us to add
    // additional hooks for other types in the future if needed
    // without making older code, which called `relate`, obsolete.

    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>, side_effects: &mut S)
        -> RelateResult<'tcx, Ty<'tcx>>;

    fn regions(&mut self, a: ty::Region, b: ty::Region,
               side_effects: &mut S)
        -> RelateResult<'tcx, ty::Region>;

    fn binders<T>(&mut self, a: &ty::Binder<T>, b: &ty::Binder<T>,
                  side_effects: &mut S)
        -> RelateResult<'tcx, ty::Binder<T>>
        where T: Relate<'a,'tcx>;
}

pub trait Relate<'a,'tcx>: TypeFoldable<'tcx> {
    fn relate<R: TypeRelation<'a, 'tcx, S>, S>(relation: &mut R,
                                               a: &Self,
                                               b: &Self,
                                               side_effects: &mut S)
                                               -> RelateResult<'tcx, Self>;
}

///////////////////////////////////////////////////////////////////////////
// Relate impls

impl<'a, 'tcx: 'a> Relate<'a, 'tcx> for ty::TypeAndMut<'tcx> {
    fn relate<R, S>(relation: &mut R,
                    a: &ty::TypeAndMut<'tcx>,
                    b: &ty::TypeAndMut<'tcx>,
                    side_effects: &mut S)
                    -> RelateResult<'tcx, ty::TypeAndMut<'tcx>>
        where R: TypeRelation<'a, 'tcx, S>
    {
        debug!("{}.mts({:?}, {:?})",
               relation.tag(),
               a,
               b);
        if a.mutbl != b.mutbl {
            Err(TypeError::Mutability)
        } else {
            let mutbl = a.mutbl;
            let variance = match mutbl {
                ast::Mutability::MutImmutable => ty::Covariant,
                ast::Mutability::MutMutable => ty::Invariant,
            };
            let ty = relation.relate_with_variance(variance, &a.ty, &b.ty, side_effects)?;
            Ok(ty::TypeAndMut {ty: ty, mutbl: mutbl})
        }
    }
}

// substitutions are not themselves relatable without more context,
// but they is an important subroutine for things that ARE relatable,
// like traits etc.
fn relate_item_substs<'a, 'tcx, R, S>(relation: &mut R,
                                      item_def_id: DefId,
                                      a_subst: &Substs<'tcx>,
                                      b_subst: &Substs<'tcx>,
                                      side_effects: &mut S)
                                      -> RelateResult<'tcx, Substs<'tcx>>
    where R: TypeRelation<'a, 'tcx, S>, 'tcx: 'a
{
    debug!("substs: item_def_id={:?} a_subst={:?} b_subst={:?}",
           item_def_id,
           a_subst,
           b_subst);

    let variances;
    let opt_variances = if relation.tcx().variance_computed.get() {
        variances = relation.tcx().item_variances(item_def_id);
        Some(&*variances)
    } else {
        None
    };
    relate_substs(relation, opt_variances, a_subst, b_subst, side_effects)
}

pub fn relate_substs<'a, 'tcx, R, S>(relation: &mut R,
                                     variances: Option<&ty::ItemVariances>,
                                     a_subst: &Substs<'tcx>,
                                     b_subst: &Substs<'tcx>,
                                     side_effects: &mut S)
                                     -> RelateResult<'tcx, Substs<'tcx>>
    where R: TypeRelation<'a, 'tcx, S>, 'tcx: 'a
{
    let mut substs = Substs::empty();

    for &space in &ParamSpace::all() {
        let a_tps = a_subst.types.get_slice(space);
        let b_tps = b_subst.types.get_slice(space);
        let t_variances = variances.map(|v| v.types.get_slice(space));
        let tps = relate_type_params(relation, t_variances, a_tps, b_tps, side_effects)?;
        substs.types.replace(space, tps);
    }

    for &space in &ParamSpace::all() {
        let a_regions = a_subst.regions.get_slice(space);
        let b_regions = b_subst.regions.get_slice(space);
        let r_variances = variances.map(|v| v.regions.get_slice(space));
        let regions = relate_region_params(relation,
                                           r_variances,
                                           a_regions,
                                           b_regions,
                                           side_effects)?;
        substs.regions.replace(space, regions);
    }

    Ok(substs)
}

fn relate_type_params<'a, 'tcx, R, S>(relation: &mut R,
                                      variances: Option<&[ty::Variance]>,
                                      a_tys: &[Ty<'tcx>],
                                      b_tys: &[Ty<'tcx>],
                                      side_effects: &mut S)
                                      -> RelateResult<'tcx, Vec<Ty<'tcx>>>
    where R: TypeRelation<'a, 'tcx, S>, 'tcx: 'a
{
    if a_tys.len() != b_tys.len() {
        return Err(TypeError::TyParamSize(expected_found(relation,
                                                         &a_tys.len(),
                                                         &b_tys.len())));
    }

    (0 .. a_tys.len())
        .map(|i| {
            let a_ty = a_tys[i];
            let b_ty = b_tys[i];
            let v = variances.map_or(ty::Invariant, |v| v[i]);
            relation.relate_with_variance(v, &a_ty, &b_ty, side_effects)
        })
        .collect()
}

fn relate_region_params<'a, 'tcx, R, S>(relation: &mut R,
                                        variances: Option<&[ty::Variance]>,
                                        a_rs: &[ty::Region],
                                        b_rs: &[ty::Region],
                                        side_effects: &mut S)
                                        -> RelateResult<'tcx, Vec<ty::Region>>
    where R: TypeRelation<'a, 'tcx, S>, 'tcx: 'a
{
    let num_region_params = a_rs.len();

    debug!("relate_region_params(a_rs={:?}, \
            b_rs={:?}, variances={:?})",
           a_rs,
           b_rs,
           variances);

    assert_eq!(num_region_params,
               variances.map_or(num_region_params,
                                |v| v.len()));

    assert_eq!(num_region_params, b_rs.len());

    (0..a_rs.len())
        .map(|i| {
            let a_r = a_rs[i];
            let b_r = b_rs[i];
            let variance = variances.map_or(ty::Invariant, |v| v[i]);
            relation.relate_with_variance(variance, &a_r, &b_r, side_effects)
        })
        .collect()
}

impl<'a, 'tcx: 'a> Relate<'a, 'tcx> for ty::BareFnTy<'tcx> {
    fn relate<R, S>(relation: &mut R,
                    a: &ty::BareFnTy<'tcx>,
                    b: &ty::BareFnTy<'tcx>,
                    side_effects: &mut S)
                    -> RelateResult<'tcx, ty::BareFnTy<'tcx>>
        where R: TypeRelation<'a, 'tcx, S>
    {
        let unsafety = relation.relate(&a.unsafety, &b.unsafety, side_effects)?;
        let abi = relation.relate(&a.abi, &b.abi, side_effects)?;
        let sig = relation.relate(&a.sig, &b.sig, side_effects)?;
        Ok(ty::BareFnTy {unsafety: unsafety,
                         abi: abi,
                         sig: sig})
    }
}

impl<'a, 'tcx: 'a> Relate<'a, 'tcx> for ty::FnSig<'tcx> {
    fn relate<R, S>(relation: &mut R,
                    a: &ty::FnSig<'tcx>,
                    b: &ty::FnSig<'tcx>,
                    side_effects: &mut S)
                    -> RelateResult<'tcx, ty::FnSig<'tcx>>
        where R: TypeRelation<'a, 'tcx, S>
    {
        if a.variadic != b.variadic {
            return Err(TypeError::VariadicMismatch(
                expected_found(relation, &a.variadic, &b.variadic)));
        }

        let inputs = relate_arg_vecs(relation,
                                     &a.inputs,
                                     &b.inputs,
                                     side_effects)?;

        let output = match (a.output, b.output) {
            (ty::FnConverging(a_ty), ty::FnConverging(b_ty)) =>
                Ok(ty::FnConverging(relation.relate(&a_ty, &b_ty, side_effects)?)),
            (ty::FnDiverging, ty::FnDiverging) =>
                Ok(ty::FnDiverging),
            (a, b) =>
                Err(TypeError::ConvergenceMismatch(
                    expected_found(relation, &(a != ty::FnDiverging), &(b != ty::FnDiverging)))),
        }?;

        return Ok(ty::FnSig {inputs: inputs,
                             output: output,
                             variadic: a.variadic});
    }
}

fn relate_arg_vecs<'a, 'tcx, R, S>(relation: &mut R,
                                   a_args: &[Ty<'tcx>],
                                   b_args: &[Ty<'tcx>],
                                   side_effects: &mut S)
                                   -> RelateResult<'tcx, Vec<Ty<'tcx>>>
    where R: TypeRelation<'a, 'tcx, S>, 'tcx: 'a
{
    if a_args.len() != b_args.len() {
        return Err(TypeError::ArgCount);
    }

    a_args.iter().zip(b_args)
          .map(|(a, b)| relation.relate_with_variance(ty::Contravariant, a, b, side_effects))
          .collect()
}

impl<'a, 'tcx: 'a> Relate<'a, 'tcx> for ast::Unsafety {
    fn relate<R, S>(relation: &mut R,
                   a: &ast::Unsafety,
                   b: &ast::Unsafety,
                   _: &mut S)
                   -> RelateResult<'tcx, ast::Unsafety>
        where R: TypeRelation<'a, 'tcx, S>
    {
        if a != b {
            Err(TypeError::UnsafetyMismatch(expected_found(relation, a, b)))
        } else {
            Ok(*a)
        }
    }
}

impl<'a, 'tcx: 'a> Relate<'a, 'tcx> for abi::Abi {
    fn relate<R, S>(relation: &mut R,
                    a: &abi::Abi,
                    b: &abi::Abi,
                    _: &mut S)
                    -> RelateResult<'tcx, abi::Abi>
        where R: TypeRelation<'a, 'tcx, S>
    {
        if a == b {
            Ok(*a)
        } else {
            Err(TypeError::AbiMismatch(expected_found(relation, a, b)))
        }
    }
}

impl<'a,'tcx: 'a> Relate<'a, 'tcx> for ty::ProjectionTy<'tcx> {
    fn relate<R, S>(relation: &mut R,
                    a: &ty::ProjectionTy<'tcx>,
                    b: &ty::ProjectionTy<'tcx>,
                    side_effects: &mut S)
                    -> RelateResult<'tcx, ty::ProjectionTy<'tcx>>
        where R: TypeRelation<'a, 'tcx, S>
    {
        if a.item_name != b.item_name {
            Err(TypeError::ProjectionNameMismatched(
                expected_found(relation, &a.item_name, &b.item_name)))
        } else {
            let trait_ref = relation.relate(&a.trait_ref, &b.trait_ref, side_effects)?;
            Ok(ty::ProjectionTy { trait_ref: trait_ref, item_name: a.item_name })
        }
    }
}

impl<'a, 'tcx: 'a> Relate<'a, 'tcx> for ty::ProjectionPredicate<'tcx> {
    fn relate<R, S>(relation: &mut R,
                    a: &ty::ProjectionPredicate<'tcx>,
                    b: &ty::ProjectionPredicate<'tcx>,
                    side_effects: &mut S)
                    -> RelateResult<'tcx, ty::ProjectionPredicate<'tcx>>
        where R: TypeRelation<'a, 'tcx, S>
    {
        let projection_ty = relation.relate(&a.projection_ty, &b.projection_ty, side_effects)?;
        let ty = relation.relate(&a.ty, &b.ty, side_effects)?;
        Ok(ty::ProjectionPredicate { projection_ty: projection_ty, ty: ty })
    }
}

impl<'a, 'tcx: 'a> Relate<'a, 'tcx> for Vec<ty::PolyProjectionPredicate<'tcx>> {
    fn relate<R, S>(relation: &mut R,
                    a: &Vec<ty::PolyProjectionPredicate<'tcx>>,
                    b: &Vec<ty::PolyProjectionPredicate<'tcx>>,
                    side_effects: &mut S)
                    -> RelateResult<'tcx, Vec<ty::PolyProjectionPredicate<'tcx>>>
        where R: TypeRelation<'a, 'tcx, S>
    {
        // To be compatible, `a` and `b` must be for precisely the
        // same set of traits and item names. We always require that
        // projection bounds lists are sorted by trait-def-id and item-name,
        // so we can just iterate through the lists pairwise, so long as they are the
        // same length.
        if a.len() != b.len() {
            Err(TypeError::ProjectionBoundsLength(expected_found(relation, &a.len(), &b.len())))
        } else {
            a.iter().zip(b)
                .map(|(a, b)| relation.relate(a, b, side_effects))
                .collect()
        }
    }
}

impl<'a, 'tcx: 'a> Relate<'a, 'tcx> for ty::ExistentialBounds<'tcx> {
    fn relate<R, S>(relation: &mut R,
                    a: &ty::ExistentialBounds<'tcx>,
                    b: &ty::ExistentialBounds<'tcx>,
                    side_effects: &mut S)
                    -> RelateResult<'tcx, ty::ExistentialBounds<'tcx>>
        where R: TypeRelation<'a, 'tcx, S>
    {
        let r =
            relation.with_cause(
                Cause::ExistentialRegionBound,
                |relation| relation.relate_with_variance(ty::Contravariant,
                                                         &a.region_bound,
                                                         &b.region_bound,
                                                         side_effects))?;
        let nb = relation.relate(&a.builtin_bounds, &b.builtin_bounds, side_effects)?;
        let pb = relation.relate(&a.projection_bounds, &b.projection_bounds, side_effects)?;
        Ok(ty::ExistentialBounds { region_bound: r,
                                   builtin_bounds: nb,
                                   projection_bounds: pb })
    }
}

impl<'a, 'tcx: 'a> Relate<'a, 'tcx> for ty::BuiltinBounds {
    fn relate<R, S>(relation: &mut R,
                    a: &ty::BuiltinBounds,
                    b: &ty::BuiltinBounds,
                    _: &mut S)
                    -> RelateResult<'tcx, ty::BuiltinBounds>
        where R: TypeRelation<'a, 'tcx, S>
    {
        // Two sets of builtin bounds are only relatable if they are
        // precisely the same (but see the coercion code).
        if a != b {
            Err(TypeError::BuiltinBoundsMismatch(expected_found(relation, a, b)))
        } else {
            Ok(*a)
        }
    }
}

impl<'a, 'tcx: 'a> Relate<'a, 'tcx> for ty::TraitRef<'tcx> {
    fn relate<R, S>(relation: &mut R,
                    a: &ty::TraitRef<'tcx>,
                    b: &ty::TraitRef<'tcx>,
                    side_effects: &mut S)
                    -> RelateResult<'tcx, ty::TraitRef<'tcx>>
        where R: TypeRelation<'a, 'tcx, S>
    {
        // Different traits cannot be related
        if a.def_id != b.def_id {
            Err(TypeError::Traits(expected_found(relation, &a.def_id, &b.def_id)))
        } else {
            let substs = relate_item_substs(relation, a.def_id, a.substs, b.substs, side_effects)?;
            Ok(ty::TraitRef { def_id: a.def_id, substs: relation.tcx().mk_substs(substs) })
        }
    }
}

impl<'a, 'tcx: 'a> Relate<'a, 'tcx> for Ty<'tcx> {
    fn relate<R, S>(relation: &mut R,
                    a: &Ty<'tcx>,
                    b: &Ty<'tcx>,
                    side_effects: &mut S)
                    -> RelateResult<'tcx, Ty<'tcx>>
        where R: TypeRelation<'a, 'tcx, S>
    {
        relation.tys(a, b, side_effects)
    }
}

/// The main "type relation" routine. Note that this does not handle
/// inference artifacts, so you should filter those out before calling
/// it.
pub fn super_relate_tys<'a, 'tcx, R, S>(relation: &mut R,
                                        a: Ty<'tcx>,
                                        b: Ty<'tcx>,
                                        side_effects: &mut S)
                                        -> RelateResult<'tcx, Ty<'tcx>>
    where R: TypeRelation<'a, 'tcx, S>, 'tcx: 'a
{
    let tcx = relation.tcx();
    let a_sty = &a.sty;
    let b_sty = &b.sty;
    debug!("super_tys: a_sty={:?} b_sty={:?}", a_sty, b_sty);
    match (a_sty, b_sty) {
        (&ty::TyInfer(_), _) |
        (_, &ty::TyInfer(_)) =>
        {
            // The caller should handle these cases!
            tcx.sess.bug("var types encountered in super_relate_tys")
        }

        (&ty::TyError, _) | (_, &ty::TyError) =>
        {
            Ok(tcx.types.err)
        }

        (&ty::TyChar, _) |
        (&ty::TyBool, _) |
        (&ty::TyInt(_), _) |
        (&ty::TyUint(_), _) |
        (&ty::TyFloat(_), _) |
        (&ty::TyStr, _)
            if a == b =>
        {
            Ok(a)
        }

        (&ty::TyParam(ref a_p), &ty::TyParam(ref b_p))
            if a_p.idx == b_p.idx && a_p.space == b_p.space =>
        {
            Ok(a)
        }

        (&ty::TyEnum(a_def, a_substs), &ty::TyEnum(b_def, b_substs))
            if a_def == b_def =>
        {
            let substs = relate_item_substs(relation,
                                            a_def.did,
                                            a_substs,
                                            b_substs,
                                            side_effects)?;
            Ok(tcx.mk_enum(a_def, tcx.mk_substs(substs)))
        }

        (&ty::TyTrait(ref a_), &ty::TyTrait(ref b_)) =>
        {
            let principal = relation.relate(&a_.principal, &b_.principal, side_effects)?;
            let bounds = relation.relate(&a_.bounds, &b_.bounds, side_effects)?;
            Ok(tcx.mk_trait(principal, bounds))
        }

        (&ty::TyStruct(a_def, a_substs), &ty::TyStruct(b_def, b_substs))
            if a_def == b_def =>
        {
            let substs = relate_item_substs(relation,
                                            a_def.did,
                                            a_substs,
                                            b_substs,
                                            side_effects)?;
            Ok(tcx.mk_struct(a_def, tcx.mk_substs(substs)))
        }

        (&ty::TyClosure(a_id, ref a_substs),
         &ty::TyClosure(b_id, ref b_substs))
            if a_id == b_id =>
        {
            // All TyClosure types with the same id represent
            // the (anonymous) type of the same closure expression. So
            // all of their regions should be equated.
            let substs = relation.relate(a_substs, b_substs, side_effects)?;
            Ok(tcx.mk_closure_from_closure_substs(a_id, substs))
        }

        (&ty::TyBox(a_inner), &ty::TyBox(b_inner)) =>
        {
            let typ = relation.relate(&a_inner, &b_inner, side_effects)?;
            Ok(tcx.mk_box(typ))
        }

        (&ty::TyRawPtr(ref a_mt), &ty::TyRawPtr(ref b_mt)) =>
        {
            let mt = relation.relate(a_mt, b_mt, side_effects)?;
            Ok(tcx.mk_ptr(mt))
        }

        (&ty::TyRef(a_r, ref a_mt), &ty::TyRef(b_r, ref b_mt)) =>
        {
            let r = relation.relate_with_variance(ty::Contravariant,
                                                  a_r,
                                                  b_r,
                                                  side_effects)?;
            let mt = relation.relate(a_mt, b_mt, side_effects)?;
            Ok(tcx.mk_ref(tcx.mk_region(r), mt))
        }

        (&ty::TyArray(a_t, sz_a), &ty::TyArray(b_t, sz_b)) =>
        {
            let t = relation.relate(&a_t, &b_t, side_effects)?;
            if sz_a == sz_b {
                Ok(tcx.mk_array(t, sz_a))
            } else {
                Err(TypeError::FixedArraySize(expected_found(relation, &sz_a, &sz_b)))
            }
        }

        (&ty::TySlice(a_t), &ty::TySlice(b_t)) =>
        {
            let t = relation.relate(&a_t, &b_t, side_effects)?;
            Ok(tcx.mk_slice(t))
        }

        (&ty::TyTuple(ref as_), &ty::TyTuple(ref bs)) =>
        {
            if as_.len() == bs.len() {
                let ts = as_.iter().zip(bs)
                            .map(|(a, b)| relation.relate(a, b, side_effects))
                            .collect::<Result<_, _>>()?;
                Ok(tcx.mk_tup(ts))
            } else if !(as_.is_empty() || bs.is_empty()) {
                Err(TypeError::TupleSize(
                    expected_found(relation, &as_.len(), &bs.len())))
            } else {
                Err(TypeError::Sorts(expected_found(relation, &a, &b)))
            }
        }

        (&ty::TyFnDef(a_def_id, a_substs, a_fty),
         &ty::TyFnDef(b_def_id, b_substs, b_fty))
            if a_def_id == b_def_id =>
        {
            let substs = relate_substs(relation, None, a_substs, b_substs, side_effects)?;
            let fty = relation.relate(a_fty, b_fty, side_effects)?;
            Ok(tcx.mk_fn_def(a_def_id, tcx.mk_substs(substs), fty))
        }

        (&ty::TyFnPtr(a_fty), &ty::TyFnPtr(b_fty)) =>
        {
            let fty = relation.relate(a_fty, b_fty, side_effects)?;
            Ok(tcx.mk_fn_ptr(fty))
        }

        (&ty::TyProjection(ref a_data), &ty::TyProjection(ref b_data)) =>
        {
            let projection_ty = relation.relate(a_data, b_data, side_effects)?;
            Ok(tcx.mk_projection(projection_ty.trait_ref, projection_ty.item_name))
        }

        _ =>
        {
            Err(TypeError::Sorts(expected_found(relation, &a, &b)))
        }
    }
}

impl<'a, 'tcx: 'a> Relate<'a, 'tcx> for ty::ClosureSubsts<'tcx> {
    fn relate<R, S>(relation: &mut R,
                    a: &ty::ClosureSubsts<'tcx>,
                    b: &ty::ClosureSubsts<'tcx>,
                    side_effects: &mut S)
                    -> RelateResult<'tcx, ty::ClosureSubsts<'tcx>>
        where R: TypeRelation<'a, 'tcx, S>
    {
        let func_substs = relate_substs(relation,
                                        None,
                                        a.func_substs,
                                        b.func_substs,
                                        side_effects)?;
        let upvar_tys = relation.relate_zip(&a.upvar_tys, &b.upvar_tys, side_effects)?;
        Ok(ty::ClosureSubsts { func_substs: relation.tcx().mk_substs(func_substs),
                               upvar_tys: upvar_tys })
    }
}

impl<'a, 'tcx: 'a> Relate<'a, 'tcx> for ty::Region {
    fn relate<R, S>(relation: &mut R,
                   a: &ty::Region,
                   b: &ty::Region,
                   side_effects: &mut S)
                   -> RelateResult<'tcx, ty::Region>
        where R: TypeRelation<'a, 'tcx, S>
    {
        relation.regions(*a, *b, side_effects)
    }
}

impl<'a, 'tcx: 'a, T> Relate<'a, 'tcx> for ty::Binder<T>
    where T: Relate<'a,'tcx>
{
    fn relate<R, S>(relation: &mut R,
                    a: &ty::Binder<T>,
                    b: &ty::Binder<T>,
                    side_effects: &mut S)
                    -> RelateResult<'tcx, ty::Binder<T>>
        where R: TypeRelation<'a, 'tcx, S>
    {
        relation.binders(a, b, side_effects)
    }
}

impl<'a, 'tcx:'a, T> Relate<'a, 'tcx> for Rc<T>
    where T: Relate<'a,'tcx>
{
    fn relate<R, S>(relation: &mut R,
                    a: &Rc<T>,
                    b: &Rc<T>,
                    side_effects: &mut S)
                    -> RelateResult<'tcx, Rc<T>>
        where R: TypeRelation<'a, 'tcx, S>
    {
        let a: &T = a;
        let b: &T = b;
        Ok(Rc::new(relation.relate(a, b, side_effects)?))
    }
}

impl<'a, 'tcx: 'a, T> Relate<'a, 'tcx> for Box<T>
    where T: Relate<'a,'tcx>
{
    fn relate<R, S>(relation: &mut R,
                    a: &Box<T>,
                    b: &Box<T>,
                    side_effects: &mut S)
                    -> RelateResult<'tcx, Box<T>>
        where R: TypeRelation<'a, 'tcx, S>
    {
        let a: &T = a;
        let b: &T = b;
        Ok(Box::new(relation.relate(a, b, side_effects)?))
    }
}

///////////////////////////////////////////////////////////////////////////
// Error handling

pub fn expected_found<'a, 'tcx: 'a, R, S, T>(relation: &mut R,
                                             a: &T,
                                             b: &T)
                                             -> ExpectedFound<T>
    where R: TypeRelation<'a, 'tcx, S>, T: Clone
{
    expected_found_bool(relation.a_is_expected(), a, b)
}

pub fn expected_found_bool<T>(a_is_expected: bool,
                              a: &T,
                              b: &T)
                              -> ExpectedFound<T>
    where T: Clone
{
    let a = a.clone();
    let b = b.clone();
    if a_is_expected {
        ExpectedFound {expected: a, found: b}
    } else {
        ExpectedFound {expected: b, found: a}
    }
}
