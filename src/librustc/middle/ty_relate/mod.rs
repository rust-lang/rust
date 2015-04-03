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

use middle::subst::{ErasedRegions, NonerasedRegions, ParamSpace, Substs};
use middle::ty::{self, Ty};
use middle::ty_fold::TypeFoldable;
use std::rc::Rc;
use syntax::abi;
use syntax::ast;
use util::ppaux::Repr;

pub type RelateResult<'tcx, T> = Result<T, ty::type_err<'tcx>>;

pub trait TypeRelation<'a,'tcx> : Sized {
    fn tcx(&self) -> &'a ty::ctxt<'tcx>;

    /// Returns a static string we can use for printouts.
    fn tag(&self) -> &'static str;

    /// Returns true if the value `a` is the "expected" type in the
    /// relation. Just affects error messages.
    fn a_is_expected(&self) -> bool;

    /// Generic relation routine suitable for most anything.
    fn relate<T:Relate<'a,'tcx>>(&mut self, a: &T, b: &T) -> RelateResult<'tcx, T> {
        Relate::relate(self, a, b)
    }

    /// Switch variance for the purpose of relating `a` and `b`.
    fn relate_with_variance<T:Relate<'a,'tcx>>(&mut self,
                                               variance: ty::Variance,
                                               a: &T,
                                               b: &T)
                                               -> RelateResult<'tcx, T>;

    // Overrideable relations. You shouldn't typically call these
    // directly, instead call `relate()`, which in turn calls
    // these. This is both more uniform but also allows us to add
    // additional hooks for other types in the future if needed
    // without making older code, which called `relate`, obsolete.

    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>)
           -> RelateResult<'tcx, Ty<'tcx>>;

    fn regions(&mut self, a: ty::Region, b: ty::Region)
               -> RelateResult<'tcx, ty::Region>;

    fn binders<T>(&mut self, a: &ty::Binder<T>, b: &ty::Binder<T>)
                  -> RelateResult<'tcx, ty::Binder<T>>
        where T: Relate<'a,'tcx>;
}

pub trait Relate<'a,'tcx>: TypeFoldable<'tcx> {
    fn relate<R:TypeRelation<'a,'tcx>>(relation: &mut R,
                                       a: &Self,
                                       b: &Self)
                                       -> RelateResult<'tcx, Self>;
}

///////////////////////////////////////////////////////////////////////////
// Relate impls

impl<'a,'tcx:'a> Relate<'a,'tcx> for ty::mt<'tcx> {
    fn relate<R>(relation: &mut R,
                 a: &ty::mt<'tcx>,
                 b: &ty::mt<'tcx>)
                 -> RelateResult<'tcx, ty::mt<'tcx>>
        where R: TypeRelation<'a,'tcx>
    {
        debug!("{}.mts({}, {})",
               relation.tag(),
               a.repr(relation.tcx()),
               b.repr(relation.tcx()));
        if a.mutbl != b.mutbl {
            Err(ty::terr_mutability)
        } else {
            let mutbl = a.mutbl;
            let variance = match mutbl {
                ast::MutImmutable => ty::Covariant,
                ast::MutMutable => ty::Invariant,
            };
            let ty = try!(relation.relate_with_variance(variance, &a.ty, &b.ty));
            Ok(ty::mt {ty: ty, mutbl: mutbl})
        }
    }
}

// substitutions are not themselves relatable without more context,
// but they is an important subroutine for things that ARE relatable,
// like traits etc.
fn relate_item_substs<'a,'tcx:'a,R>(relation: &mut R,
                                    item_def_id: ast::DefId,
                                    a_subst: &Substs<'tcx>,
                                    b_subst: &Substs<'tcx>)
                                    -> RelateResult<'tcx, Substs<'tcx>>
    where R: TypeRelation<'a,'tcx>
{
    debug!("substs: item_def_id={} a_subst={} b_subst={}",
           item_def_id.repr(relation.tcx()),
           a_subst.repr(relation.tcx()),
           b_subst.repr(relation.tcx()));

    let variances;
    let opt_variances = if relation.tcx().variance_computed.get() {
        variances = ty::item_variances(relation.tcx(), item_def_id);
        Some(&*variances)
    } else {
        None
    };
    relate_substs(relation, opt_variances, a_subst, b_subst)
}

fn relate_substs<'a,'tcx:'a,R>(relation: &mut R,
                               variances: Option<&ty::ItemVariances>,
                               a_subst: &Substs<'tcx>,
                               b_subst: &Substs<'tcx>)
                               -> RelateResult<'tcx, Substs<'tcx>>
    where R: TypeRelation<'a,'tcx>
{
    let mut substs = Substs::empty();

    for &space in &ParamSpace::all() {
        let a_tps = a_subst.types.get_slice(space);
        let b_tps = b_subst.types.get_slice(space);
        let t_variances = variances.map(|v| v.types.get_slice(space));
        let tps = try!(relate_type_params(relation, t_variances, a_tps, b_tps));
        substs.types.replace(space, tps);
    }

    match (&a_subst.regions, &b_subst.regions) {
        (&ErasedRegions, _) | (_, &ErasedRegions) => {
            substs.regions = ErasedRegions;
        }

        (&NonerasedRegions(ref a), &NonerasedRegions(ref b)) => {
            for &space in &ParamSpace::all() {
                let a_regions = a.get_slice(space);
                let b_regions = b.get_slice(space);
                let r_variances = variances.map(|v| v.regions.get_slice(space));
                let regions = try!(relate_region_params(relation,
                                                        r_variances,
                                                        a_regions,
                                                        b_regions));
                substs.mut_regions().replace(space, regions);
            }
        }
    }

    Ok(substs)
}

fn relate_type_params<'a,'tcx:'a,R>(relation: &mut R,
                                    variances: Option<&[ty::Variance]>,
                                    a_tys: &[Ty<'tcx>],
                                    b_tys: &[Ty<'tcx>])
                                    -> RelateResult<'tcx, Vec<Ty<'tcx>>>
    where R: TypeRelation<'a,'tcx>
{
    if a_tys.len() != b_tys.len() {
        return Err(ty::terr_ty_param_size(expected_found(relation,
                                                         &a_tys.len(),
                                                         &b_tys.len())));
    }

    (0 .. a_tys.len())
        .map(|i| {
            let a_ty = a_tys[i];
            let b_ty = b_tys[i];
            let v = variances.map_or(ty::Invariant, |v| v[i]);
            relation.relate_with_variance(v, &a_ty, &b_ty)
        })
        .collect()
}

fn relate_region_params<'a,'tcx:'a,R>(relation: &mut R,
                                      variances: Option<&[ty::Variance]>,
                                      a_rs: &[ty::Region],
                                      b_rs: &[ty::Region])
                                      -> RelateResult<'tcx, Vec<ty::Region>>
    where R: TypeRelation<'a,'tcx>
{
    let tcx = relation.tcx();
    let num_region_params = a_rs.len();

    debug!("relate_region_params(a_rs={}, \
            b_rs={}, variances={})",
           a_rs.repr(tcx),
           b_rs.repr(tcx),
           variances.repr(tcx));

    assert_eq!(num_region_params,
               variances.map_or(num_region_params,
                                |v| v.len()));

    assert_eq!(num_region_params, b_rs.len());

    (0..a_rs.len())
        .map(|i| {
            let a_r = a_rs[i];
            let b_r = b_rs[i];
            let variance = variances.map_or(ty::Invariant, |v| v[i]);
            relation.relate_with_variance(variance, &a_r, &b_r)
        })
        .collect()
}

impl<'a,'tcx:'a> Relate<'a,'tcx> for ty::BareFnTy<'tcx> {
    fn relate<R>(relation: &mut R,
                 a: &ty::BareFnTy<'tcx>,
                 b: &ty::BareFnTy<'tcx>)
                 -> RelateResult<'tcx, ty::BareFnTy<'tcx>>
        where R: TypeRelation<'a,'tcx>
    {
        let unsafety = try!(relation.relate(&a.unsafety, &b.unsafety));
        let abi = try!(relation.relate(&a.abi, &b.abi));
        let sig = try!(relation.relate(&a.sig, &b.sig));
        Ok(ty::BareFnTy {unsafety: unsafety,
                         abi: abi,
                         sig: sig})
    }
}

impl<'a,'tcx:'a> Relate<'a,'tcx> for ty::FnSig<'tcx> {
    fn relate<R>(relation: &mut R,
                 a: &ty::FnSig<'tcx>,
                 b: &ty::FnSig<'tcx>)
                 -> RelateResult<'tcx, ty::FnSig<'tcx>>
        where R: TypeRelation<'a,'tcx>
    {
        if a.variadic != b.variadic {
            return Err(ty::terr_variadic_mismatch(
                expected_found(relation, &a.variadic, &b.variadic)));
        }

        let inputs = try!(relate_arg_vecs(relation,
                                          &a.inputs,
                                          &b.inputs));

        let output = try!(match (a.output, b.output) {
            (ty::FnConverging(a_ty), ty::FnConverging(b_ty)) =>
                Ok(ty::FnConverging(try!(relation.relate(&a_ty, &b_ty)))),
            (ty::FnDiverging, ty::FnDiverging) =>
                Ok(ty::FnDiverging),
            (a, b) =>
                Err(ty::terr_convergence_mismatch(
                    expected_found(relation, &(a != ty::FnDiverging), &(b != ty::FnDiverging)))),
        });

        return Ok(ty::FnSig {inputs: inputs,
                             output: output,
                             variadic: a.variadic});
    }
}

fn relate_arg_vecs<'a,'tcx:'a,R>(relation: &mut R,
                                 a_args: &[Ty<'tcx>],
                                 b_args: &[Ty<'tcx>])
                                 -> RelateResult<'tcx, Vec<Ty<'tcx>>>
    where R: TypeRelation<'a,'tcx>
{
    if a_args.len() != b_args.len() {
        return Err(ty::terr_arg_count);
    }

    a_args.iter()
          .zip(b_args.iter())
          .map(|(a, b)| relation.relate_with_variance(ty::Contravariant, a, b))
          .collect()
}

impl<'a,'tcx:'a> Relate<'a,'tcx> for ast::Unsafety {
    fn relate<R>(relation: &mut R,
                 a: &ast::Unsafety,
                 b: &ast::Unsafety)
                 -> RelateResult<'tcx, ast::Unsafety>
        where R: TypeRelation<'a,'tcx>
    {
        if a != b {
            Err(ty::terr_unsafety_mismatch(expected_found(relation, a, b)))
        } else {
            Ok(*a)
        }
    }
}

impl<'a,'tcx:'a> Relate<'a,'tcx> for abi::Abi {
    fn relate<R>(relation: &mut R,
                 a: &abi::Abi,
                 b: &abi::Abi)
                 -> RelateResult<'tcx, abi::Abi>
        where R: TypeRelation<'a,'tcx>
    {
        if a == b {
            Ok(*a)
        } else {
            Err(ty::terr_abi_mismatch(expected_found(relation, a, b)))
        }
    }
}

impl<'a,'tcx:'a> Relate<'a,'tcx> for ty::ProjectionTy<'tcx> {
    fn relate<R>(relation: &mut R,
                 a: &ty::ProjectionTy<'tcx>,
                 b: &ty::ProjectionTy<'tcx>)
                 -> RelateResult<'tcx, ty::ProjectionTy<'tcx>>
        where R: TypeRelation<'a,'tcx>
    {
        if a.item_name != b.item_name {
            Err(ty::terr_projection_name_mismatched(
                expected_found(relation, &a.item_name, &b.item_name)))
        } else {
            let trait_ref = try!(relation.relate(&*a.trait_ref, &*b.trait_ref));
            Ok(ty::ProjectionTy { trait_ref: Rc::new(trait_ref), item_name: a.item_name })
        }
    }
}

impl<'a,'tcx:'a> Relate<'a,'tcx> for ty::ProjectionPredicate<'tcx> {
    fn relate<R>(relation: &mut R,
                 a: &ty::ProjectionPredicate<'tcx>,
                 b: &ty::ProjectionPredicate<'tcx>)
                 -> RelateResult<'tcx, ty::ProjectionPredicate<'tcx>>
        where R: TypeRelation<'a,'tcx>
    {
        let projection_ty = try!(relation.relate(&a.projection_ty, &b.projection_ty));
        let ty = try!(relation.relate(&a.ty, &b.ty));
        Ok(ty::ProjectionPredicate { projection_ty: projection_ty, ty: ty })
    }
}

impl<'a,'tcx:'a> Relate<'a,'tcx> for Vec<ty::PolyProjectionPredicate<'tcx>> {
    fn relate<R>(relation: &mut R,
                 a: &Vec<ty::PolyProjectionPredicate<'tcx>>,
                 b: &Vec<ty::PolyProjectionPredicate<'tcx>>)
                 -> RelateResult<'tcx, Vec<ty::PolyProjectionPredicate<'tcx>>>
        where R: TypeRelation<'a,'tcx>
    {
        // To be compatible, `a` and `b` must be for precisely the
        // same set of traits and item names. We always require that
        // projection bounds lists are sorted by trait-def-id and item-name,
        // so we can just iterate through the lists pairwise, so long as they are the
        // same length.
        if a.len() != b.len() {
            Err(ty::terr_projection_bounds_length(expected_found(relation, &a.len(), &b.len())))
        } else {
            a.iter()
                .zip(b.iter())
                .map(|(a, b)| relation.relate(a, b))
                .collect()
        }
    }
}

impl<'a,'tcx:'a> Relate<'a,'tcx> for ty::ExistentialBounds<'tcx> {
    fn relate<R>(relation: &mut R,
                 a: &ty::ExistentialBounds<'tcx>,
                 b: &ty::ExistentialBounds<'tcx>)
                 -> RelateResult<'tcx, ty::ExistentialBounds<'tcx>>
        where R: TypeRelation<'a,'tcx>
    {
        let r = try!(relation.relate_with_variance(ty::Contravariant,
                                                   &a.region_bound,
                                                   &b.region_bound));
        let nb = try!(relation.relate(&a.builtin_bounds, &b.builtin_bounds));
        let pb = try!(relation.relate(&a.projection_bounds, &b.projection_bounds));
        Ok(ty::ExistentialBounds { region_bound: r,
                                   builtin_bounds: nb,
                                   projection_bounds: pb })
    }
}

impl<'a,'tcx:'a> Relate<'a,'tcx> for ty::BuiltinBounds {
    fn relate<R>(relation: &mut R,
                 a: &ty::BuiltinBounds,
                 b: &ty::BuiltinBounds)
                 -> RelateResult<'tcx, ty::BuiltinBounds>
        where R: TypeRelation<'a,'tcx>
    {
        // Two sets of builtin bounds are only relatable if they are
        // precisely the same (but see the coercion code).
        if a != b {
            Err(ty::terr_builtin_bounds(expected_found(relation, a, b)))
        } else {
            Ok(*a)
        }
    }
}

impl<'a,'tcx:'a> Relate<'a,'tcx> for ty::TraitRef<'tcx> {
    fn relate<R>(relation: &mut R,
                 a: &ty::TraitRef<'tcx>,
                 b: &ty::TraitRef<'tcx>)
                 -> RelateResult<'tcx, ty::TraitRef<'tcx>>
        where R: TypeRelation<'a,'tcx>
    {
        // Different traits cannot be related
        if a.def_id != b.def_id {
            Err(ty::terr_traits(expected_found(relation, &a.def_id, &b.def_id)))
        } else {
            let substs = try!(relate_item_substs(relation, a.def_id, a.substs, b.substs));
            Ok(ty::TraitRef { def_id: a.def_id, substs: relation.tcx().mk_substs(substs) })
        }
    }
}

impl<'a,'tcx:'a> Relate<'a,'tcx> for Ty<'tcx> {
    fn relate<R>(relation: &mut R,
                 a: &Ty<'tcx>,
                 b: &Ty<'tcx>)
                 -> RelateResult<'tcx, Ty<'tcx>>
        where R: TypeRelation<'a,'tcx>
    {
        relation.tys(a, b)
    }
}

/// The main "type relation" routine. Note that this does not handle
/// inference artifacts, so you should filter those out before calling
/// it.
pub fn super_relate_tys<'a,'tcx:'a,R>(relation: &mut R,
                                      a: Ty<'tcx>,
                                      b: Ty<'tcx>)
                                      -> RelateResult<'tcx, Ty<'tcx>>
    where R: TypeRelation<'a,'tcx>
{
    let tcx = relation.tcx();
    let a_sty = &a.sty;
    let b_sty = &b.sty;
    debug!("super_tys: a_sty={:?} b_sty={:?}", a_sty, b_sty);
    match (a_sty, b_sty) {
        (&ty::ty_infer(_), _) |
        (_, &ty::ty_infer(_)) =>
        {
            // The caller should handle these cases!
            tcx.sess.bug("var types encountered in super_relate_tys")
        }

        (&ty::ty_err, _) | (_, &ty::ty_err) =>
        {
            Ok(tcx.types.err)
        }

        (&ty::ty_char, _) |
        (&ty::ty_bool, _) |
        (&ty::ty_int(_), _) |
        (&ty::ty_uint(_), _) |
        (&ty::ty_float(_), _) |
        (&ty::ty_str, _)
            if a == b =>
        {
            Ok(a)
        }

        (&ty::ty_param(ref a_p), &ty::ty_param(ref b_p))
            if a_p.idx == b_p.idx && a_p.space == b_p.space =>
        {
            Ok(a)
        }

        (&ty::ty_enum(a_id, a_substs), &ty::ty_enum(b_id, b_substs))
            if a_id == b_id =>
        {
            let substs = try!(relate_item_substs(relation, a_id, a_substs, b_substs));
            Ok(ty::mk_enum(tcx, a_id, tcx.mk_substs(substs)))
        }

        (&ty::ty_trait(ref a_), &ty::ty_trait(ref b_)) =>
        {
            let principal = try!(relation.relate(&a_.principal, &b_.principal));
            let bounds = try!(relation.relate(&a_.bounds, &b_.bounds));
            Ok(ty::mk_trait(tcx, principal, bounds))
        }

        (&ty::ty_struct(a_id, a_substs), &ty::ty_struct(b_id, b_substs))
            if a_id == b_id =>
        {
            let substs = try!(relate_item_substs(relation, a_id, a_substs, b_substs));
            Ok(ty::mk_struct(tcx, a_id, tcx.mk_substs(substs)))
        }

        (&ty::ty_closure(a_id, a_substs),
         &ty::ty_closure(b_id, b_substs))
            if a_id == b_id =>
        {
            // All ty_closure types with the same id represent
            // the (anonymous) type of the same closure expression. So
            // all of their regions should be equated.
            let substs = try!(relate_substs(relation, None, a_substs, b_substs));
            Ok(ty::mk_closure(tcx, a_id, tcx.mk_substs(substs)))
        }

        (&ty::ty_uniq(a_inner), &ty::ty_uniq(b_inner)) =>
        {
            let typ = try!(relation.relate(&a_inner, &b_inner));
            Ok(ty::mk_uniq(tcx, typ))
        }

        (&ty::ty_ptr(ref a_mt), &ty::ty_ptr(ref b_mt)) =>
        {
            let mt = try!(relation.relate(a_mt, b_mt));
            Ok(ty::mk_ptr(tcx, mt))
        }

        (&ty::ty_rptr(a_r, ref a_mt), &ty::ty_rptr(b_r, ref b_mt)) =>
        {
            let r = try!(relation.relate_with_variance(ty::Contravariant, a_r, b_r));
            let mt = try!(relation.relate(a_mt, b_mt));
            Ok(ty::mk_rptr(tcx, tcx.mk_region(r), mt))
        }

        (&ty::ty_vec(a_t, Some(sz_a)), &ty::ty_vec(b_t, Some(sz_b))) =>
        {
            let t = try!(relation.relate(&a_t, &b_t));
            if sz_a == sz_b {
                Ok(ty::mk_vec(tcx, t, Some(sz_a)))
            } else {
                Err(ty::terr_fixed_array_size(expected_found(relation, &sz_a, &sz_b)))
            }
        }

        (&ty::ty_vec(a_t, None), &ty::ty_vec(b_t, None)) =>
        {
            let t = try!(relation.relate(&a_t, &b_t));
            Ok(ty::mk_vec(tcx, t, None))
        }

        (&ty::ty_tup(ref as_), &ty::ty_tup(ref bs)) =>
        {
            if as_.len() == bs.len() {
                let ts = try!(as_.iter()
                                 .zip(bs.iter())
                                 .map(|(a, b)| relation.relate(a, b))
                                 .collect::<Result<_, _>>());
                Ok(ty::mk_tup(tcx, ts))
            } else if as_.len() != 0 && bs.len() != 0 {
                Err(ty::terr_tuple_size(
                    expected_found(relation, &as_.len(), &bs.len())))
            } else {
                Err(ty::terr_sorts(expected_found(relation, &a, &b)))
            }
        }

        (&ty::ty_bare_fn(a_opt_def_id, a_fty), &ty::ty_bare_fn(b_opt_def_id, b_fty))
            if a_opt_def_id == b_opt_def_id =>
        {
            let fty = try!(relation.relate(a_fty, b_fty));
            Ok(ty::mk_bare_fn(tcx, a_opt_def_id, tcx.mk_bare_fn(fty)))
        }

        (&ty::ty_projection(ref a_data), &ty::ty_projection(ref b_data)) =>
        {
            let projection_ty = try!(relation.relate(a_data, b_data));
            Ok(ty::mk_projection(tcx, projection_ty.trait_ref, projection_ty.item_name))
        }

        _ =>
        {
            Err(ty::terr_sorts(expected_found(relation, &a, &b)))
        }
    }
}

impl<'a,'tcx:'a> Relate<'a,'tcx> for ty::Region {
    fn relate<R>(relation: &mut R,
                 a: &ty::Region,
                 b: &ty::Region)
                 -> RelateResult<'tcx, ty::Region>
        where R: TypeRelation<'a,'tcx>
    {
        relation.regions(*a, *b)
    }
}

impl<'a,'tcx:'a,T> Relate<'a,'tcx> for ty::Binder<T>
    where T: Relate<'a,'tcx>
{
    fn relate<R>(relation: &mut R,
                 a: &ty::Binder<T>,
                 b: &ty::Binder<T>)
                 -> RelateResult<'tcx, ty::Binder<T>>
        where R: TypeRelation<'a,'tcx>
    {
        relation.binders(a, b)
    }
}

impl<'a,'tcx:'a,T> Relate<'a,'tcx> for Rc<T>
    where T: Relate<'a,'tcx>
{
    fn relate<R>(relation: &mut R,
                 a: &Rc<T>,
                 b: &Rc<T>)
                 -> RelateResult<'tcx, Rc<T>>
        where R: TypeRelation<'a,'tcx>
    {
        let a: &T = a;
        let b: &T = b;
        Ok(Rc::new(try!(relation.relate(a, b))))
    }
}

impl<'a,'tcx:'a,T> Relate<'a,'tcx> for Box<T>
    where T: Relate<'a,'tcx>
{
    fn relate<R>(relation: &mut R,
                 a: &Box<T>,
                 b: &Box<T>)
                 -> RelateResult<'tcx, Box<T>>
        where R: TypeRelation<'a,'tcx>
    {
        let a: &T = a;
        let b: &T = b;
        Ok(Box::new(try!(relation.relate(a, b))))
    }
}

///////////////////////////////////////////////////////////////////////////
// Error handling

pub fn expected_found<'a,'tcx:'a,R,T>(relation: &mut R,
                                      a: &T,
                                      b: &T)
                                      -> ty::expected_found<T>
    where R: TypeRelation<'a,'tcx>, T: Clone
{
    expected_found_bool(relation.a_is_expected(), a, b)
}

pub fn expected_found_bool<T>(a_is_expected: bool,
                              a: &T,
                              b: &T)
                              -> ty::expected_found<T>
    where T: Clone
{
    let a = a.clone();
    let b = b.clone();
    if a_is_expected {
        ty::expected_found {expected: a, found: b}
    } else {
        ty::expected_found {expected: b, found: a}
    }
}

