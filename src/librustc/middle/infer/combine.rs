// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

///////////////////////////////////////////////////////////////////////////
// # Type combining
//
// There are four type combiners: equate, sub, lub, and glb.  Each
// implements the trait `Combine` and contains methods for combining
// two instances of various things and yielding a new instance.  These
// combiner methods always yield a `Result<T>`.  There is a lot of
// common code for these operations, implemented as default methods on
// the `Combine` trait.
//
// Each operation may have side-effects on the inference context,
// though these can be unrolled using snapshots. On success, the
// LUB/GLB operations return the appropriate bound. The Eq and Sub
// operations generally return the first operand.
//
// ## Contravariance
//
// When you are relating two things which have a contravariant
// relationship, you should use `contratys()` or `contraregions()`,
// rather than inversing the order of arguments!  This is necessary
// because the order of arguments is not relevant for LUB and GLB.  It
// is also useful to track which value is the "expected" value in
// terms of error reporting.

use super::bivariate::Bivariate;
use super::equate::Equate;
use super::glb::Glb;
use super::lub::Lub;
use super::sub::Sub;
use super::{InferCtxt};
use super::{MiscVariable, TypeTrace};
use super::type_variable::{RelationDir, BiTo, EqTo, SubtypeOf, SupertypeOf};

use middle::ty::{IntType, UintType};
use middle::ty::{self, Ty};
use middle::ty::error::TypeError;
use middle::ty::fold::{TypeFolder, TypeFoldable};
use middle::ty::relate::{Relate, RelateResult, TypeRelation};

use syntax::ast;
use syntax::codemap::Span;

#[derive(Clone)]
pub struct CombineFields<'a, 'tcx: 'a> {
    pub infcx: &'a InferCtxt<'a, 'tcx>,
    pub a_is_expected: bool,
    pub trace: TypeTrace<'tcx>,
    pub cause: Option<ty::relate::Cause>,
}

pub fn super_combine_tys<'a,'tcx:'a,R>(infcx: &InferCtxt<'a, 'tcx>,
                                       relation: &mut R,
                                       a: Ty<'tcx>,
                                       b: Ty<'tcx>)
                                       -> RelateResult<'tcx, Ty<'tcx>>
    where R: TypeRelation<'a,'tcx>
{
    let a_is_expected = relation.a_is_expected();

    match (&a.sty, &b.sty) {
        // Relate integral variables to other types
        (&ty::TyInfer(ty::IntVar(a_id)), &ty::TyInfer(ty::IntVar(b_id))) => {
            try!(infcx.int_unification_table
                      .borrow_mut()
                      .unify_var_var(a_id, b_id)
                      .map_err(|e| int_unification_error(a_is_expected, e)));
            Ok(a)
        }
        (&ty::TyInfer(ty::IntVar(v_id)), &ty::TyInt(v)) => {
            unify_integral_variable(infcx, a_is_expected, v_id, IntType(v))
        }
        (&ty::TyInt(v), &ty::TyInfer(ty::IntVar(v_id))) => {
            unify_integral_variable(infcx, !a_is_expected, v_id, IntType(v))
        }
        (&ty::TyInfer(ty::IntVar(v_id)), &ty::TyUint(v)) => {
            unify_integral_variable(infcx, a_is_expected, v_id, UintType(v))
        }
        (&ty::TyUint(v), &ty::TyInfer(ty::IntVar(v_id))) => {
            unify_integral_variable(infcx, !a_is_expected, v_id, UintType(v))
        }

        // Relate floating-point variables to other types
        (&ty::TyInfer(ty::FloatVar(a_id)), &ty::TyInfer(ty::FloatVar(b_id))) => {
            try!(infcx.float_unification_table
                      .borrow_mut()
                      .unify_var_var(a_id, b_id)
                      .map_err(|e| float_unification_error(relation.a_is_expected(), e)));
            Ok(a)
        }
        (&ty::TyInfer(ty::FloatVar(v_id)), &ty::TyFloat(v)) => {
            unify_float_variable(infcx, a_is_expected, v_id, v)
        }
        (&ty::TyFloat(v), &ty::TyInfer(ty::FloatVar(v_id))) => {
            unify_float_variable(infcx, !a_is_expected, v_id, v)
        }

        // All other cases of inference are errors
        (&ty::TyInfer(_), _) |
        (_, &ty::TyInfer(_)) => {
            Err(TypeError::Sorts(ty::relate::expected_found(relation, &a, &b)))
        }


        _ => {
            ty::relate::super_relate_tys(relation, a, b)
        }
    }
}

fn unify_integral_variable<'a,'tcx>(infcx: &InferCtxt<'a,'tcx>,
                                    vid_is_expected: bool,
                                    vid: ty::IntVid,
                                    val: ty::IntVarValue)
                                    -> RelateResult<'tcx, Ty<'tcx>>
{
    try!(infcx
         .int_unification_table
         .borrow_mut()
         .unify_var_value(vid, val)
         .map_err(|e| int_unification_error(vid_is_expected, e)));
    match val {
        IntType(v) => Ok(infcx.tcx.mk_mach_int(v)),
        UintType(v) => Ok(infcx.tcx.mk_mach_uint(v)),
    }
}

fn unify_float_variable<'a,'tcx>(infcx: &InferCtxt<'a,'tcx>,
                                 vid_is_expected: bool,
                                 vid: ty::FloatVid,
                                 val: ast::FloatTy)
                                 -> RelateResult<'tcx, Ty<'tcx>>
{
    try!(infcx
         .float_unification_table
         .borrow_mut()
         .unify_var_value(vid, val)
         .map_err(|e| float_unification_error(vid_is_expected, e)));
    Ok(infcx.tcx.mk_mach_float(val))
}

impl<'a, 'tcx> CombineFields<'a, 'tcx> {
    pub fn tcx(&self) -> &'a ty::ctxt<'tcx> {
        self.infcx.tcx
    }

    pub fn switch_expected(&self) -> CombineFields<'a, 'tcx> {
        CombineFields {
            a_is_expected: !self.a_is_expected,
            ..(*self).clone()
        }
    }

    pub fn equate(&self) -> Equate<'a, 'tcx> {
        Equate::new(self.clone())
    }

    pub fn bivariate(&self) -> Bivariate<'a, 'tcx> {
        Bivariate::new(self.clone())
    }

    pub fn sub(&self) -> Sub<'a, 'tcx> {
        Sub::new(self.clone())
    }

    pub fn lub(&self) -> Lub<'a, 'tcx> {
        Lub::new(self.clone())
    }

    pub fn glb(&self) -> Glb<'a, 'tcx> {
        Glb::new(self.clone())
    }

    pub fn instantiate(&self,
                       a_ty: Ty<'tcx>,
                       dir: RelationDir,
                       b_vid: ty::TyVid)
                       -> RelateResult<'tcx, ()>
    {
        let mut stack = Vec::new();
        stack.push((a_ty, dir, b_vid));
        loop {
            // For each turn of the loop, we extract a tuple
            //
            //     (a_ty, dir, b_vid)
            //
            // to relate. Here dir is either SubtypeOf or
            // SupertypeOf. The idea is that we should ensure that
            // the type `a_ty` is a subtype or supertype (respectively) of the
            // type to which `b_vid` is bound.
            //
            // If `b_vid` has not yet been instantiated with a type
            // (which is always true on the first iteration, but not
            // necessarily true on later iterations), we will first
            // instantiate `b_vid` with a *generalized* version of
            // `a_ty`. Generalization introduces other inference
            // variables wherever subtyping could occur (at time of
            // this writing, this means replacing free regions with
            // region variables).
            let (a_ty, dir, b_vid) = match stack.pop() {
                None => break,
                Some(e) => e,
            };

            debug!("instantiate(a_ty={:?} dir={:?} b_vid={:?})",
                   a_ty,
                   dir,
                   b_vid);

            // Check whether `vid` has been instantiated yet.  If not,
            // make a generalized form of `ty` and instantiate with
            // that.
            let b_ty = self.infcx.type_variables.borrow().probe(b_vid);
            let b_ty = match b_ty {
                Some(t) => t, // ...already instantiated.
                None => {     // ...not yet instantiated:
                    // Generalize type if necessary.
                    let generalized_ty = try!(match dir {
                        EqTo => self.generalize(a_ty, b_vid, false),
                        BiTo | SupertypeOf | SubtypeOf => self.generalize(a_ty, b_vid, true),
                    });
                    debug!("instantiate(a_ty={:?}, dir={:?}, \
                                        b_vid={:?}, generalized_ty={:?})",
                           a_ty, dir, b_vid,
                           generalized_ty);
                    self.infcx.type_variables
                        .borrow_mut()
                        .instantiate_and_push(
                            b_vid, generalized_ty, &mut stack);
                    generalized_ty
                }
            };

            // The original triple was `(a_ty, dir, b_vid)` -- now we have
            // resolved `b_vid` to `b_ty`, so apply `(a_ty, dir, b_ty)`:
            //
            // FIXME(#16847): This code is non-ideal because all these subtype
            // relations wind up attributed to the same spans. We need
            // to associate causes/spans with each of the relations in
            // the stack to get this right.
            try!(match dir {
                BiTo => self.bivariate().relate(&a_ty, &b_ty),
                EqTo => self.equate().relate(&a_ty, &b_ty),
                SubtypeOf => self.sub().relate(&a_ty, &b_ty),
                SupertypeOf => self.sub().relate_with_variance(ty::Contravariant, &a_ty, &b_ty),
            });
        }

        Ok(())
    }

    /// Attempts to generalize `ty` for the type variable `for_vid`.  This checks for cycle -- that
    /// is, whether the type `ty` references `for_vid`. If `make_region_vars` is true, it will also
    /// replace all regions with fresh variables. Returns `TyError` in the case of a cycle, `Ok`
    /// otherwise.
    fn generalize(&self,
                  ty: Ty<'tcx>,
                  for_vid: ty::TyVid,
                  make_region_vars: bool)
                  -> RelateResult<'tcx, Ty<'tcx>>
    {
        let mut generalize = Generalizer {
            infcx: self.infcx,
            span: self.trace.origin.span(),
            for_vid: for_vid,
            make_region_vars: make_region_vars,
            cycle_detected: false
        };
        let u = ty.fold_with(&mut generalize);
        if generalize.cycle_detected {
            Err(TypeError::CyclicTy)
        } else {
            Ok(u)
        }
    }
}

struct Generalizer<'cx, 'tcx:'cx> {
    infcx: &'cx InferCtxt<'cx, 'tcx>,
    span: Span,
    for_vid: ty::TyVid,
    make_region_vars: bool,
    cycle_detected: bool,
}

impl<'cx, 'tcx> ty::fold::TypeFolder<'tcx> for Generalizer<'cx, 'tcx> {
    fn tcx(&self) -> &ty::ctxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        // Check to see whether the type we are genealizing references
        // `vid`. At the same time, also update any type variables to
        // the values that they are bound to. This is needed to truly
        // check for cycles, but also just makes things readable.
        //
        // (In particular, you could have something like `$0 = Box<$1>`
        //  where `$1` has already been instantiated with `Box<$0>`)
        match t.sty {
            ty::TyInfer(ty::TyVar(vid)) => {
                if vid == self.for_vid {
                    self.cycle_detected = true;
                    self.tcx().types.err
                } else {
                    match self.infcx.type_variables.borrow().probe(vid) {
                        Some(u) => self.fold_ty(u),
                        None => t,
                    }
                }
            }
            _ => {
                t.fold_subitems_with(self)
            }
        }
    }

    fn fold_region(&mut self, r: ty::Region) -> ty::Region {
        match r {
            // Never make variables for regions bound within the type itself.
            ty::ReLateBound(..) => { return r; }

            // Early-bound regions should really have been substituted away before
            // we get to this point.
            ty::ReEarlyBound(..) => {
                self.tcx().sess.span_bug(
                    self.span,
                    &format!("Encountered early bound region when generalizing: {:?}",
                            r));
            }

            // Always make a fresh region variable for skolemized regions;
            // the higher-ranked decision procedures rely on this.
            ty::ReSkolemized(..) => { }

            // For anything else, we make a region variable, unless we
            // are *equating*, in which case it's just wasteful.
            ty::ReEmpty |
            ty::ReStatic |
            ty::ReScope(..) |
            ty::ReVar(..) |
            ty::ReFree(..) => {
                if !self.make_region_vars {
                    return r;
                }
            }
        }

        // FIXME: This is non-ideal because we don't give a
        // very descriptive origin for this region variable.
        self.infcx.next_region_var(MiscVariable(self.span))
    }
}

pub trait RelateResultCompare<'tcx, T> {
    fn compare<F>(&self, t: T, f: F) -> RelateResult<'tcx, T> where
        F: FnOnce() -> TypeError<'tcx>;
}

impl<'tcx, T:Clone + PartialEq> RelateResultCompare<'tcx, T> for RelateResult<'tcx, T> {
    fn compare<F>(&self, t: T, f: F) -> RelateResult<'tcx, T> where
        F: FnOnce() -> TypeError<'tcx>,
    {
        self.clone().and_then(|s| {
            if s == t {
                self.clone()
            } else {
                Err(f())
            }
        })
    }
}

fn int_unification_error<'tcx>(a_is_expected: bool, v: (ty::IntVarValue, ty::IntVarValue))
                               -> TypeError<'tcx>
{
    let (a, b) = v;
    TypeError::IntMismatch(ty::relate::expected_found_bool(a_is_expected, &a, &b))
}

fn float_unification_error<'tcx>(a_is_expected: bool,
                                 v: (ast::FloatTy, ast::FloatTy))
                                 -> TypeError<'tcx>
{
    let (a, b) = v;
    TypeError::FloatMismatch(ty::relate::expected_found_bool(a_is_expected, &a, &b))
}
