// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Skolemization is the process of replacing unknown variables with
 * fresh types. The idea is that the type, after skolemization,
 * contains no inference variables but instead contains either a value
 * for each variable or fresh "arbitrary" types wherever a variable
 * would have been.
 *
 * Skolemization is used primarily to get a good type for inserting
 * into a cache. The result summarizes what the type inferencer knows
 * "so far". The primary place it is used right now is in the trait
 * matching algorithm, which needs to be able to cache whether an
 * `impl` self type matches some other type X -- *without* affecting
 * `X`. That means if that if the type `X` is in fact an unbound type
 * variable, we want the match to be regarded as ambiguous, because
 * depending on what type that type variable is ultimately assigned,
 * the match may or may not succeed.
 *
 * Note that you should be careful not to allow the output of
 * skolemization to leak to the user in error messages or in any other
 * form. Skolemization is only really useful as an internal detail.
 *
 * __An important detail concerning regions.__ The skolemizer also
 * replaces *all* regions with 'static. The reason behind this is
 * that, in general, we do not take region relationships into account
 * when making type-overloaded decisions. This is important because of
 * the design of the region inferencer, which is not based on
 * unification but rather on accumulating and then solving a set of
 * constraints. In contrast, the type inferencer assigns a value to
 * each type variable only once, and it does so as soon as it can, so
 * it is reasonable to ask what the type inferencer knows "so far".
 */

use middle::ty;
use middle::ty_fold;
use middle::ty_fold::TypeFoldable;
use middle::ty_fold::TypeFolder;
use std::collections::hashmap;

use super::InferCtxt;
use super::unify::InferCtxtMethodsForSimplyUnifiableTypes;

pub struct TypeSkolemizer<'a, 'tcx:'a> {
    infcx: &'a InferCtxt<'a, 'tcx>,
    skolemization_count: uint,
    skolemization_map: hashmap::HashMap<ty::InferTy, ty::t>,
}

impl<'a, 'tcx> TypeSkolemizer<'a, 'tcx> {
    pub fn new<'tcx>(infcx: &'a InferCtxt<'a, 'tcx>) -> TypeSkolemizer<'a, 'tcx> {
        TypeSkolemizer {
            infcx: infcx,
            skolemization_count: 0,
            skolemization_map: hashmap::HashMap::new(),
        }
    }

    fn skolemize(&mut self,
                 opt_ty: Option<ty::t>,
                 key: ty::InferTy,
                 skolemizer: |uint| -> ty::InferTy)
                 -> ty::t
    {
        match opt_ty {
            Some(ty) => { return ty.fold_with(self); }
            None => { }
        }

        match self.skolemization_map.entry(key) {
            hashmap::Occupied(entry) => *entry.get(),
            hashmap::Vacant(entry) => {
                let index = self.skolemization_count;
                self.skolemization_count += 1;
                let t = ty::mk_infer(self.infcx.tcx, skolemizer(index));
                entry.set(t);
                t
            }
        }
    }
}

impl<'a, 'tcx> TypeFolder<'tcx> for TypeSkolemizer<'a, 'tcx> {
    fn tcx<'b>(&'b self) -> &'b ty::ctxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_region(&mut self, r: ty::Region) -> ty::Region {
        match r {
            ty::ReEarlyBound(..) |
            ty::ReLateBound(..) => {
                // leave bound regions alone
                r
            }

            ty::ReStatic |
            ty::ReFree(_) |
            ty::ReScope(_) |
            ty::ReInfer(_) |
            ty::ReEmpty => {
                // replace all free regions with 'static
                ty::ReStatic
            }
        }
    }

    fn fold_ty(&mut self, t: ty::t) -> ty::t {
        match ty::get(t).sty {
            ty::ty_infer(ty::TyVar(v)) => {
                self.skolemize(self.infcx.type_variables.borrow().probe(v),
                               ty::TyVar(v),
                               ty::SkolemizedTy)
            }

            ty::ty_infer(ty::IntVar(v)) => {
                self.skolemize(self.infcx.probe_var(v),
                               ty::IntVar(v),
                               ty::SkolemizedIntTy)
            }

            ty::ty_infer(ty::FloatVar(v)) => {
                self.skolemize(self.infcx.probe_var(v),
                               ty::FloatVar(v),
                               ty::SkolemizedIntTy)
            }

            ty::ty_infer(ty::SkolemizedTy(c)) |
            ty::ty_infer(ty::SkolemizedIntTy(c)) => {
                if c >= self.skolemization_count {
                    self.tcx().sess.bug(
                        format!("Encountered a skolemized type with id {} \
                                 but our counter is only at {}",
                                c,
                                self.skolemization_count).as_slice());
                }
                t
            }

            ty::ty_open(..) => {
                self.tcx().sess.bug("Cannot skolemize an open existential type");
            }

            ty::ty_nil |
            ty::ty_bot |
            ty::ty_bool |
            ty::ty_char |
            ty::ty_int(..) |
            ty::ty_uint(..) |
            ty::ty_float(..) |
            ty::ty_enum(..) |
            ty::ty_uniq(..) |
            ty::ty_str |
            ty::ty_err |
            ty::ty_vec(..) |
            ty::ty_ptr(..) |
            ty::ty_rptr(..) |
            ty::ty_bare_fn(..) |
            ty::ty_closure(..) |
            ty::ty_trait(..) |
            ty::ty_struct(..) |
            ty::ty_unboxed_closure(..) |
            ty::ty_tup(..) |
            ty::ty_param(..) => {
                ty_fold::super_fold_ty(self, t)
            }
        }
    }
}
