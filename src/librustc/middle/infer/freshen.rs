// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Freshening is the process of replacing unknown variables with fresh types. The idea is that
//! the type, after freshening, contains no inference variables but instead contains either a
//! value for each variable or fresh "arbitrary" types wherever a variable would have been.
//!
//! Freshening is used primarily to get a good type for inserting into a cache. The result
//! summarizes what the type inferencer knows "so far". The primary place it is used right now is
//! in the trait matching algorithm, which needs to be able to cache whether an `impl` self type
//! matches some other type X -- *without* affecting `X`. That means if that if the type `X` is in
//! fact an unbound type variable, we want the match to be regarded as ambiguous, because depending
//! on what type that type variable is ultimately assigned, the match may or may not succeed.
//!
//! Note that you should be careful not to allow the output of freshening to leak to the user in
//! error messages or in any other form. Freshening is only really useful as an internal detail.
//!
//! __An important detail concerning regions.__ The freshener also replaces *all* regions with
//! 'static. The reason behind this is that, in general, we do not take region relationships into
//! account when making type-overloaded decisions. This is important because of the design of the
//! region inferencer, which is not based on unification but rather on accumulating and then
//! solving a set of constraints. In contrast, the type inferencer assigns a value to each type
//! variable only once, and it does so as soon as it can, so it is reasonable to ask what the type
//! inferencer knows "so far".

use middle::ty::{self, Ty, HasTypeFlags};
use middle::ty::FRESHEN_CACHE_SIZE;
use middle::ty::fold::TypeFoldable;
use middle::ty::fold::TypeFolder;
use std::collections::hash_map::{self, Entry};

use super::InferCtxt;
use super::unify_key::ToType;

pub const TYPE_VAR : usize = 0;
pub const INT_VAR : usize = 1;
pub const FLOAT_VAR : usize = 2;

pub struct TypeFreshener<'a, 'tcx:'a> {
    infcx: &'a InferCtxt<'a, 'tcx>,
    freshen_map: hash_map::HashMap<Ty<'tcx>, Ty<'tcx>>,

    fresh_counts: [u32; 3],
    fresh_vars: [[Option<Ty<'tcx>>; FRESHEN_CACHE_SIZE]; 3],
}

impl<'a, 'tcx> TypeFreshener<'a, 'tcx> {
    pub fn new(infcx: &'a InferCtxt<'a, 'tcx>) -> TypeFreshener<'a, 'tcx> {
        TypeFreshener {
            infcx: infcx,
            fresh_counts: [0; 3],
            freshen_map: hash_map::HashMap::new(),
            fresh_vars: [[None; FRESHEN_CACHE_SIZE]; 3],
        }
    }

    fn freshen<F>(&mut self,
                  opt_ty: Option<Ty<'tcx>>,
                  key: Ty<'tcx>,
                  freshener: F,
                  vt: usize)
                  -> Ty<'tcx> where
        F: FnOnce(u32) -> ty::InferTy,
    {
        match opt_ty {
            Some(ty) => { return ty.fold_with(self); }
            None => { }
        }

        let tcx = self.infcx.tcx;

        if !self.freshen_map.is_empty() {
            return match self.freshen_map.entry(key) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let index = self.fresh_counts[vt];
                    self.fresh_counts[vt] += 1;
                    let t = tcx.mk_infer(freshener(index));
                    entry.insert(t);
                    t
                }
            };
        }

        for i in 0..FRESHEN_CACHE_SIZE {
            if self.fresh_vars[vt][i] == Some(key) {
                return tcx.types.fresh_vars[vt][i];
            }
        }

        if self.fresh_counts[vt] == FRESHEN_CACHE_SIZE as u32 {
            debug!("freshen: cache overflow");
            self.copy_vars_to_table();
            self.freshen(None, key, freshener, vt)
        } else {
            self.fresh_vars[vt][self.fresh_counts[vt] as usize]
                = Some(key);
            let result = tcx.types.fresh_vars[vt][self.fresh_counts[vt] as usize];
            self.fresh_counts[vt] += 1;
            result
        }
    }

    pub fn copy_vars_to_table(&mut self) {
        for vt in 0..2 {
            for i in 0..FRESHEN_CACHE_SIZE {
                if let Some(t) = self.fresh_vars[vt][i] {
                    self.freshen_map.insert(
                        t, self.infcx.tcx.types.fresh_vars[vt][i]);
                }
            }
        }
    }
}

impl<'a, 'tcx> TypeFolder<'tcx> for TypeFreshener<'a, 'tcx> {
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
            ty::ReVar(_) |
            ty::ReSkolemized(..) |
            ty::ReEmpty => {
                // replace all free regions with 'static
                ty::ReStatic
            }
        }
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if !t.needs_infer() && !t.has_erasable_regions() {
            return t;
        }

        let tcx = self.infcx.tcx;

        match t.sty {
            ty::TyInfer(ty::TyVar(v)) => {
                self.freshen(
                    self.infcx.type_variables.borrow().probe(v),
                    t,
                    ty::FreshTy,
                    TYPE_VAR)
            }

            ty::TyInfer(ty::IntVar(v)) => {
                self.freshen(
                    self.infcx.int_unification_table.borrow_mut()
                                                    .probe(v)
                                                    .map(|v| v.to_type(tcx)),
                    t,
                    ty::FreshIntTy,
                    INT_VAR)
            }

            ty::TyInfer(ty::FloatVar(v)) => {
                self.freshen(
                    self.infcx.float_unification_table.borrow_mut()
                                                      .probe(v)
                                                      .map(|v| v.to_type(tcx)),
                    t,
                    ty::FreshFloatTy,
                    FLOAT_VAR)
            }

            ty::TyInfer(ty::FreshTy(c)) => {
                if c >= self.fresh_counts[TYPE_VAR] {
                    tcx.sess.bug(
                        &format!("Encountered a freshend type with id {} \
                                  but our counter is only at {}",
                                 c,
                                 self.fresh_counts[TYPE_VAR]));
                }
                t
            }

            ty::TyInfer(ty::FreshIntTy(c)) => {
                if c >= self.fresh_counts[INT_VAR] {
                    tcx.sess.bug(
                        &format!("Encountered a freshend int type with id {} \
                                  but our counter is only at {}",
                                 c,
                                 self.fresh_counts[INT_VAR]));
                }
                t
            }

            ty::TyInfer(ty::FreshFloatTy(c)) => {
                if c >= self.fresh_counts[FLOAT_VAR] {
                    tcx.sess.bug(
                        &format!("Encountered a freshend float type with id {} \
                                  but our counter is only at {}",
                                 c,
                                 self.fresh_counts[FLOAT_VAR]));
                }
                t
            }

            ty::TyBool |
            ty::TyChar |
            ty::TyInt(..) |
            ty::TyUint(..) |
            ty::TyFloat(..) |
            ty::TyEnum(..) |
            ty::TyBox(..) |
            ty::TyStr |
            ty::TyError |
            ty::TyArray(..) |
            ty::TySlice(..) |
            ty::TyRawPtr(..) |
            ty::TyRef(..) |
            ty::TyBareFn(..) |
            ty::TyTrait(..) |
            ty::TyStruct(..) |
            ty::TyClosure(..) |
            ty::TyTuple(..) |
            ty::TyProjection(..) |
            ty::TyParam(..) => {
                ty::fold::super_fold_ty(self, t)
            }
        }
    }
}
