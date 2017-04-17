// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ty::{self, Ty, TyCtxt};
use util::common::MemoizationMap;
use util::nodemap::FxHashMap;

bitflags! {
    /// Type contents is how the type checker reasons about kinds.
    /// They track what kinds of things are found within a type.  You can
    /// think of them as kind of an "anti-kind".  They track the kinds of values
    /// and thinks that are contained in types.  Having a larger contents for
    /// a type tends to rule that type *out* from various kinds.  For example,
    /// a type that contains a reference is not sendable.
    ///
    /// The reason we compute type contents and not kinds is that it is
    /// easier for me (nmatsakis) to think about what is contained within
    /// a type than to think about what is *not* contained within a type.
    flags TypeContents: u8 {
        const OWNS_DTOR         = 0b1,
    }
}

impl TypeContents {
    pub fn when(&self, cond: bool) -> TypeContents {
        if cond {*self} else {TypeContents::empty()}
    }

    pub fn needs_drop(&self, _: TyCtxt) -> bool {
        self.intersects(TypeContents::OWNS_DTOR)
    }

    pub fn union<I, T, F>(v: I, mut f: F) -> TypeContents where
        I: IntoIterator<Item=T>,
        F: FnMut(T) -> TypeContents,
    {
        v.into_iter().fold(TypeContents::empty(), |tc, ty| tc | f(ty))
    }
}

impl<'a, 'tcx> ty::TyS<'tcx> {
    pub fn type_contents(&'tcx self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> TypeContents {
        return tcx.tc_cache.memoize(self, || tc_ty(tcx, self, &mut FxHashMap()));

        fn tc_ty<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                           ty: Ty<'tcx>,
                           cache: &mut FxHashMap<Ty<'tcx>, TypeContents>) -> TypeContents
        {
            // Subtle: Note that we are *not* using tcx.tc_cache here but rather a
            // private cache for this walk.  This is needed in the case of cyclic
            // types like:
            //
            //     struct List { next: Box<Option<List>>, ... }
            //
            // When computing the type contents of such a type, we wind up deeply
            // recursing as we go.  So when we encounter the recursive reference
            // to List, we temporarily use TypeContents::empty() as its contents.  Later we'll
            // patch up the cache with the correct value, once we've computed it
            // (this is basically a co-inductive process, if that helps).  So in
            // the end we'll compute TypeContents::OwnsOwned, in this case.
            //
            // The problem is, as we are doing the computation, we will also
            // compute an *intermediate* contents for, e.g., Option<List> of
            // TypeContents::empty().  This is ok during the computation of List itself, but if
            // we stored this intermediate value into tcx.tc_cache, then later
            // requests for the contents of Option<List> would also yield TypeContents::empty()
            // which is incorrect.  This value was computed based on the crutch
            // value for the type contents of list.  The correct value is
            // TypeContents::OwnsOwned.  This manifested as issue #4821.
            if let Some(tc) = cache.get(&ty) {
                return *tc;
            }
            // Must check both caches!
            if let Some(tc) = tcx.tc_cache.borrow().get(&ty) {
                return *tc;
            }
            cache.insert(ty, TypeContents::empty());

            let result = match ty.sty {
                ty::TyInfer(ty::FreshIntTy(_)) | ty::TyInfer(ty::FreshFloatTy(_)) |
                ty::TyBool | ty::TyInt(_) | ty::TyUint(_) | ty::TyFloat(_) | ty::TyNever |
                ty::TyFnDef(..) | ty::TyFnPtr(_) | ty::TyChar |
                ty::TyRawPtr(_) | ty::TyRef(..) |
                ty::TyStr => TypeContents::empty(),

                ty::TyArray(ty, _) => {
                    tc_ty(tcx, ty, cache)
                }

                ty::TySlice(ty) => {
                    tc_ty(tcx, ty, cache)
                }

                ty::TyClosure(def_id, ref substs) => {
                    TypeContents::union(
                        substs.upvar_tys(def_id, tcx),
                        |ty| tc_ty(tcx, &ty, cache))
                }

                ty::TyTuple(ref tys, _) => {
                    TypeContents::union(&tys[..],
                                        |ty| tc_ty(tcx, *ty, cache))
                }

                ty::TyAdt(def, substs) => {
                    TypeContents::union(&def.variants, |v| {
                        TypeContents::union(&v.fields, |f| {
                            tc_ty(tcx, f.ty(tcx, substs), cache)
                        })
                    })

                    // unions don't have destructors regardless of the child types
                        - TypeContents::OWNS_DTOR.when(def.is_union())
                        | TypeContents::OWNS_DTOR.when(def.has_dtor(tcx))
                }

                ty::TyDynamic(..) |
                ty::TyProjection(..) |
                ty::TyParam(_) |
                ty::TyAnon(..) => TypeContents::OWNS_DTOR,

                ty::TyInfer(_) |
                ty::TyError => {
                    bug!("asked to compute contents of error type");
                }
            };

            cache.insert(ty, result);
            result
        }
    }
}
