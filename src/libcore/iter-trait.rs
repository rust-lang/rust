// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This makes use of a clever hack that brson came up with to
// workaround our lack of traits and lack of macros.  See core.{rc,rs} for
// how this file is used.

use cmp::{Eq, Ord};
use iter::BaseIter;
use iter;
use kinds::Copy;
use option::Option;

use self::inst::{IMPL_T, EACH, SIZE_HINT};

impl<A> iter::BaseIter<A> for IMPL_T<A> {
    #[inline(always)]
    pure fn each(&self, blk: fn(v: &A) -> bool) { EACH(self, blk) }
    #[inline(always)]
    pure fn size_hint(&self) -> Option<uint> { SIZE_HINT(self) }
}

impl<A> iter::ExtendedIter<A> for IMPL_T<A> {
    #[inline(always)]
    pure fn eachi(&self, blk: fn(uint, v: &A) -> bool) {
        iter::eachi(self, blk)
    }
    #[inline(always)]
    pure fn all(&self, blk: fn(&A) -> bool) -> bool {
        iter::all(self, blk)
    }
    #[inline(always)]
    pure fn any(&self, blk: fn(&A) -> bool) -> bool {
        iter::any(self, blk)
    }
    #[inline(always)]
    pure fn foldl<B>(&self, b0: B, blk: fn(&B, &A) -> B) -> B {
        iter::foldl(self, move b0, blk)
    }
    #[inline(always)]
    pure fn position(&self, f: fn(&A) -> bool) -> Option<uint> {
        iter::position(self, f)
    }
    #[inline(always)]
    pure fn map_to_vec<B>(&self, op: fn(&A) -> B) -> ~[B] {
        iter::map_to_vec(self, op)
    }
    #[inline(always)]
    pure fn flat_map_to_vec<B,IB:BaseIter<B>>(&self, op: fn(&A) -> IB)
        -> ~[B] {
        iter::flat_map_to_vec(self, op)
    }

}

impl<A: Eq> iter::EqIter<A> for IMPL_T<A> {
    #[inline(always)]
    pure fn contains(&self, x: &A) -> bool { iter::contains(self, x) }
    #[inline(always)]
    pure fn count(&self, x: &A) -> uint { iter::count(self, x) }
}

impl<A: Copy> iter::CopyableIter<A> for IMPL_T<A> {
    #[inline(always)]
    pure fn filter_to_vec(&self, pred: fn(&A) -> bool) -> ~[A] {
        iter::filter_to_vec(self, pred)
    }
    #[inline(always)]
    pure fn to_vec(&self) -> ~[A] { iter::to_vec(self) }
    #[inline(always)]
    pure fn find(&self, f: fn(&A) -> bool) -> Option<A> {
        iter::find(self, f)
    }
}

impl<A: Copy Ord> iter::CopyableOrderedIter<A> for IMPL_T<A> {
    #[inline(always)]
    pure fn min(&self) -> A { iter::min(self) }
    #[inline(always)]
    pure fn max(&self) -> A { iter::max(self) }
}

