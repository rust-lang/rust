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
//! To handle closures, freshened types also have to contain the signature and kind of any
//! closure in the local inference context, as otherwise the cache key might be invalidated.
//! The way this is done is somewhat hacky - the closure signature is appended to the substs,
//! as well as the closure kind "encoded" as a type. Also, special handling is needed when
//! the closure signature contains a reference to the original closure.
//!
//! Note that you should be careful not to allow the output of freshening to leak to the user in
//! error messages or in any other form. Freshening is only really useful as an internal detail.
//!
//! Because of the manipulation required to handle closures, doing arbitrary operations on
//! freshened types is not recommended. However, in addition to doing equality/hash
//! comparisons (for caching), it is possible to do a `ty::_match` operation between
//! 2 freshened types - this works even with the closure encoding.
//!
//! __An important detail concerning regions.__ The freshener also replaces *all* free regions with
//! 'erased. The reason behind this is that, in general, we do not take region relationships into
//! account when making type-overloaded decisions. This is important because of the design of the
//! region inferencer, which is not based on unification but rather on accumulating and then
//! solving a set of constraints. In contrast, the type inferencer assigns a value to each type
//! variable only once, and it does so as soon as it can, so it is reasonable to ask what the type
//! inferencer knows "so far".

use ty::{self, Ty, TyCtxt, TypeFoldable};
use ty::fold::TypeFolder;
use ty::subst::Substs;
use util::nodemap::FxHashMap;
use hir::def_id::DefId;

use std::collections::hash_map::Entry;

use super::InferCtxt;
use super::unify_key::ToType;

pub struct TypeFreshener<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
    freshen_count: u32,
    freshen_map: FxHashMap<ty::InferTy, Ty<'tcx>>,
    closure_set: Vec<DefId>,
}

impl<'a, 'gcx, 'tcx> TypeFreshener<'a, 'gcx, 'tcx> {
    pub fn new(infcx: &'a InferCtxt<'a, 'gcx, 'tcx>)
               -> TypeFreshener<'a, 'gcx, 'tcx> {
        TypeFreshener {
            infcx,
            freshen_count: 0,
            freshen_map: FxHashMap(),
            closure_set: vec![],
        }
    }

    fn freshen<F>(&mut self,
                  opt_ty: Option<Ty<'tcx>>,
                  key: ty::InferTy,
                  freshener: F)
                  -> Ty<'tcx> where
        F: FnOnce(u32) -> ty::InferTy,
    {
        if let Some(ty) = opt_ty {
            return ty.fold_with(self);
        }

        match self.freshen_map.entry(key) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let index = self.freshen_count;
                self.freshen_count += 1;
                let t = self.infcx.tcx.mk_infer(freshener(index));
                entry.insert(t);
                t
            }
        }
    }

    fn next_fresh<F>(&mut self,
                     freshener: F)
                     -> Ty<'tcx>
        where F: FnOnce(u32) -> ty::InferTy,
    {
        let index = self.freshen_count;
        self.freshen_count += 1;
        self.infcx.tcx.mk_infer(freshener(index))
    }

    fn freshen_closure_like<M, C>(&mut self,
                                  def_id: DefId,
                                  substs: ty::ClosureSubsts<'tcx>,
                                  t: Ty<'tcx>,
                                  markers: M,
                                  combine: C)
                                  -> Ty<'tcx>
        where M: FnOnce(&mut Self) -> (Ty<'tcx>, Ty<'tcx>),
              C: FnOnce(&'tcx Substs<'tcx>) -> Ty<'tcx>
    {
        let tcx = self.infcx.tcx;

        let closure_in_progress = self.infcx.in_progress_tables.map_or(false, |tables| {
            tcx.hir.as_local_node_id(def_id).map_or(false, |closure_id| {
                tables.borrow().local_id_root ==
                    Some(DefId::local(tcx.hir.node_to_hir_id(closure_id).owner))
            })
        });

        if !closure_in_progress {
            // If this closure belongs to another infcx, its kind etc. were
            // fully inferred and its signature/kind are exactly what's listed
            // in its infcx. So we don't need to add the markers for them.
            return t.super_fold_with(self);
        }

        // We are encoding a closure in progress. Because we want our freshening
        // key to contain all inference information needed to make sense of our
        // value, we need to encode the closure signature and kind. The way
        // we do that is to add them as 2 variables to the closure substs,
        // basically because it's there (and nobody cares about adding extra stuff
        // to substs).
        //
        // This means the "freshened" closure substs ends up looking like
        //     fresh_substs = [PARENT_SUBSTS* ; UPVARS* ; SIG_MARKER ; KIND_MARKER]
        let (marker_1, marker_2) = if self.closure_set.contains(&def_id) {
            // We found the closure def-id within its own signature. Just
            // leave a new freshened type - any matching operations would
            // have found and compared the exterior closure already to
            // get here.
            //
            // In that case, we already know what the signature would
            // be - the parent closure on the stack already contains a
            // "copy" of the signature, so there is no reason to encode
            // it again for injectivity. Just use a fresh type variable
            // to make everything comparable.
            //
            // For example (closure kinds omitted for clarity)
            //     t=[closure FOO sig=[closure BAR sig=[closure FOO ..]]]
            // Would get encoded to
            //     t=[closure FOO sig=[closure BAR sig=[closure FOO sig=$0]]]
            //
            // and we can decode by having
            //     $0=[closure BAR {sig doesn't exist in decode}]
            // and get
            //     t=[closure FOO]
            //     sig[FOO] = [closure BAR]
            //     sig[BAR] = [closure FOO]
            (self.next_fresh(ty::FreshTy), self.next_fresh(ty::FreshTy))
        } else {
            self.closure_set.push(def_id);
            let markers = markers(self);
            self.closure_set.pop();
            markers
        };

        combine(tcx.mk_substs(
            substs.substs.iter().map(|k| k.fold_with(self)).chain(
                [marker_1, marker_2].iter().cloned().map(From::from)
                    )))
    }
}

impl<'a, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for TypeFreshener<'a, 'gcx, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> {
        self.infcx.tcx
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match *r {
            ty::ReLateBound(..) => {
                // leave bound regions alone
                r
            }

            ty::ReStatic |
            ty::ReEarlyBound(..) |
            ty::ReFree(_) |
            ty::ReScope(_) |
            ty::ReVar(_) |
            ty::ReSkolemized(..) |
            ty::ReEmpty |
            ty::ReErased => {
                // replace all free regions with 'erased
                self.tcx().types.re_erased
            }
        }
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if !t.needs_infer() && !t.has_erasable_regions() &&
            !(t.has_closure_types() && self.infcx.in_progress_tables.is_some()) {
            return t;
        }

        let tcx = self.infcx.tcx;

        match t.sty {
            ty::TyInfer(ty::TyVar(v)) => {
                let opt_ty = self.infcx.type_variables.borrow_mut().probe(v);
                self.freshen(
                    opt_ty,
                    ty::TyVar(v),
                    ty::FreshTy)
            }

            ty::TyInfer(ty::IntVar(v)) => {
                self.freshen(
                    self.infcx.int_unification_table.borrow_mut()
                                                    .probe(v)
                                                    .map(|v| v.to_type(tcx)),
                    ty::IntVar(v),
                    ty::FreshIntTy)
            }

            ty::TyInfer(ty::FloatVar(v)) => {
                self.freshen(
                    self.infcx.float_unification_table.borrow_mut()
                                                      .probe(v)
                                                      .map(|v| v.to_type(tcx)),
                    ty::FloatVar(v),
                    ty::FreshFloatTy)
            }

            ty::TyInfer(ty::FreshTy(c)) |
            ty::TyInfer(ty::FreshIntTy(c)) |
            ty::TyInfer(ty::FreshFloatTy(c)) => {
                if c >= self.freshen_count {
                    bug!("Encountered a freshend type with id {} \
                          but our counter is only at {}",
                         c,
                         self.freshen_count);
                }
                t
            }

            ty::TyClosure(def_id, substs) => {
                self.freshen_closure_like(
                    def_id, substs, t,
                    |this| {
                        // HACK: use a "random" integer type to mark the kind. Because
                        // different closure kinds shouldn't get unified during
                        // selection, the "subtyping" relationship (where any kind is
                        // better than no kind) shouldn't  matter here, just that the
                        // types are different.
                        let closure_kind = this.infcx.closure_kind(def_id);
                        let closure_kind_marker = match closure_kind {
                            None => tcx.types.i8,
                            Some(ty::ClosureKind::Fn) => tcx.types.i16,
                            Some(ty::ClosureKind::FnMut) => tcx.types.i32,
                            Some(ty::ClosureKind::FnOnce) => tcx.types.i64,
                        };

                        let closure_sig = this.infcx.fn_sig(def_id);
                        (tcx.mk_fn_ptr(closure_sig.fold_with(this)),
                         closure_kind_marker)
                    },
                    |substs| tcx.mk_closure(def_id, substs)
                )
            }

            ty::TyGenerator(def_id, substs, interior) => {
                self.freshen_closure_like(
                    def_id, substs, t,
                    |this| {
                        let gen_sig = this.infcx.generator_sig(def_id).unwrap();
                        // FIXME: want to revise this strategy when generator
                        // signatures can actually contain LBRs.
                        let sig = this.tcx().no_late_bound_regions(&gen_sig)
                            .unwrap_or_else(|| {
                                bug!("late-bound regions in signature of {:?}",
                                     def_id)
                            });
                        (sig.yield_ty, sig.return_ty).fold_with(this)
                    },
                    |substs| {
                        tcx.mk_generator(def_id, ty::ClosureSubsts { substs }, interior)
                    }
                )
            }

            ty::TyBool |
            ty::TyChar |
            ty::TyInt(..) |
            ty::TyUint(..) |
            ty::TyFloat(..) |
            ty::TyAdt(..) |
            ty::TyStr |
            ty::TyError |
            ty::TyArray(..) |
            ty::TySlice(..) |
            ty::TyRawPtr(..) |
            ty::TyRef(..) |
            ty::TyFnDef(..) |
            ty::TyFnPtr(_) |
            ty::TyDynamic(..) |
            ty::TyNever |
            ty::TyTuple(..) |
            ty::TyProjection(..) |
            ty::TyParam(..) |
            ty::TyAnon(..) => {
                t.super_fold_with(self)
            }
        }
    }
}
