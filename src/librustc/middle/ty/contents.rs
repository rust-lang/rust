// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::def_id::{DefId};
use middle::ty::{self, Ty};
use util::common::MemoizationMap;
use util::nodemap::FnvHashMap;

use std::fmt;
use std::ops;

use syntax::ast;

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
#[derive(Clone, Copy)]
pub struct TypeContents {
    pub bits: u64
}

macro_rules! def_type_content_sets {
    (mod $mname:ident { $($name:ident = $bits:expr),+ }) => {
        #[allow(non_snake_case)]
        mod $mname {
            use super::TypeContents;
            $(
                #[allow(non_upper_case_globals)]
                pub const $name: TypeContents = TypeContents { bits: $bits };
             )+
        }
    }
}

def_type_content_sets! {
    mod TC {
        None                                = 0b0000_0000__0000_0000__0000,

        // Things that are interior to the value (first nibble):
        InteriorUnsafe                      = 0b0000_0000__0000_0000__0010,
        InteriorParam                       = 0b0000_0000__0000_0000__0100,
        // InteriorAll                         = 0b00000000__00000000__1111,

        // Things that are owned by the value (second and third nibbles):
        OwnsOwned                           = 0b0000_0000__0000_0001__0000,
        OwnsDtor                            = 0b0000_0000__0000_0010__0000,
        OwnsAll                             = 0b0000_0000__1111_1111__0000,

        // Things that mean drop glue is necessary
        NeedsDrop                           = 0b0000_0000__0000_0111__0000,

        // All bits
        All                                 = 0b1111_1111__1111_1111__1111
    }
}

impl TypeContents {
    pub fn when(&self, cond: bool) -> TypeContents {
        if cond {*self} else {TC::None}
    }

    pub fn intersects(&self, tc: TypeContents) -> bool {
        (self.bits & tc.bits) != 0
    }

    pub fn owns_owned(&self) -> bool {
        self.intersects(TC::OwnsOwned)
    }

    pub fn interior_param(&self) -> bool {
        self.intersects(TC::InteriorParam)
    }

    pub fn interior_unsafe(&self) -> bool {
        self.intersects(TC::InteriorUnsafe)
    }

    pub fn needs_drop(&self, _: &ty::ctxt) -> bool {
        self.intersects(TC::NeedsDrop)
    }

    /// Includes only those bits that still apply when indirected through a `Box` pointer
    pub fn owned_pointer(&self) -> TypeContents {
        TC::OwnsOwned | (*self & TC::OwnsAll)
    }

    pub fn union<T, F>(v: &[T], mut f: F) -> TypeContents where
        F: FnMut(&T) -> TypeContents,
    {
        v.iter().fold(TC::None, |tc, ty| tc | f(ty))
    }

    pub fn has_dtor(&self) -> bool {
        self.intersects(TC::OwnsDtor)
    }
}

impl ops::BitOr for TypeContents {
    type Output = TypeContents;

    fn bitor(self, other: TypeContents) -> TypeContents {
        TypeContents {bits: self.bits | other.bits}
    }
}

impl ops::BitAnd for TypeContents {
    type Output = TypeContents;

    fn bitand(self, other: TypeContents) -> TypeContents {
        TypeContents {bits: self.bits & other.bits}
    }
}

impl ops::Sub for TypeContents {
    type Output = TypeContents;

    fn sub(self, other: TypeContents) -> TypeContents {
        TypeContents {bits: self.bits & !other.bits}
    }
}

impl fmt::Debug for TypeContents {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TypeContents({:b})", self.bits)
    }
}

impl<'tcx> ty::TyS<'tcx> {
    pub fn type_contents(&'tcx self, cx: &ty::ctxt<'tcx>) -> TypeContents {
        return cx.tc_cache.memoize(self, || tc_ty(cx, self, &mut FnvHashMap()));

        fn tc_ty<'tcx>(cx: &ty::ctxt<'tcx>,
                       ty: Ty<'tcx>,
                       cache: &mut FnvHashMap<Ty<'tcx>, TypeContents>) -> TypeContents
        {
            // Subtle: Note that we are *not* using cx.tc_cache here but rather a
            // private cache for this walk.  This is needed in the case of cyclic
            // types like:
            //
            //     struct List { next: Box<Option<List>>, ... }
            //
            // When computing the type contents of such a type, we wind up deeply
            // recursing as we go.  So when we encounter the recursive reference
            // to List, we temporarily use TC::None as its contents.  Later we'll
            // patch up the cache with the correct value, once we've computed it
            // (this is basically a co-inductive process, if that helps).  So in
            // the end we'll compute TC::OwnsOwned, in this case.
            //
            // The problem is, as we are doing the computation, we will also
            // compute an *intermediate* contents for, e.g., Option<List> of
            // TC::None.  This is ok during the computation of List itself, but if
            // we stored this intermediate value into cx.tc_cache, then later
            // requests for the contents of Option<List> would also yield TC::None
            // which is incorrect.  This value was computed based on the crutch
            // value for the type contents of list.  The correct value is
            // TC::OwnsOwned.  This manifested as issue #4821.
            match cache.get(&ty) {
                Some(tc) => { return *tc; }
                None => {}
            }
            match cx.tc_cache.borrow().get(&ty) {    // Must check both caches!
                Some(tc) => { return *tc; }
                None => {}
            }
            cache.insert(ty, TC::None);

            let result = match ty.sty {
                // usize and isize are ffi-unsafe
                ty::TyUint(ast::TyUs) | ty::TyInt(ast::TyIs) => {
                    TC::None
                }

                // Scalar and unique types are sendable, and durable
                ty::TyInfer(ty::FreshIntTy(_)) | ty::TyInfer(ty::FreshFloatTy(_)) |
                ty::TyBool | ty::TyInt(_) | ty::TyUint(_) | ty::TyFloat(_) |
                ty::TyBareFn(..) | ty::TyChar => {
                    TC::None
                }

                ty::TyBox(typ) => {
                    tc_ty(cx, typ, cache).owned_pointer()
                }

                ty::TyTrait(_) => {
                    TC::All - TC::InteriorParam
                }

                ty::TyRawPtr(_) => {
                    TC::None
                }

                ty::TyRef(_, _) => {
                    TC::None
                }

                ty::TyArray(ty, _) => {
                    tc_ty(cx, ty, cache)
                }

                ty::TySlice(ty) => {
                    tc_ty(cx, ty, cache)
                }
                ty::TyStr => TC::None,

                ty::TyClosure(_, ref substs) => {
                    TypeContents::union(&substs.upvar_tys, |ty| tc_ty(cx, &ty, cache))
                }

                ty::TyTuple(ref tys) => {
                    TypeContents::union(&tys[..],
                                        |ty| tc_ty(cx, *ty, cache))
                }

                ty::TyStruct(def, substs) | ty::TyEnum(def, substs) => {
                    let mut res =
                        TypeContents::union(&def.variants, |v| {
                            TypeContents::union(&v.fields, |f| {
                                tc_ty(cx, f.ty(cx, substs), cache)
                            })
                        });

                    if def.has_dtor() {
                        res = res | TC::OwnsDtor;
                    }

                    apply_lang_items(cx, def.did, res)
                }

                ty::TyProjection(..) |
                ty::TyParam(_) => {
                    TC::All
                }

                ty::TyInfer(_) |
                ty::TyError => {
                    cx.sess.bug("asked to compute contents of error type");
                }
            };

            cache.insert(ty, result);
            result
        }

        fn apply_lang_items(cx: &ty::ctxt, did: DefId, tc: TypeContents)
                            -> TypeContents {
            if Some(did) == cx.lang_items.unsafe_cell_type() {
                tc | TC::InteriorUnsafe
            } else {
                tc
            }
        }
    }
}
