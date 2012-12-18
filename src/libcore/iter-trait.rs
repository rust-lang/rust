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

#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use cmp::{Eq, Ord};

#[cfg(stage0)]
use inst::{IMPL_T, EACH, SIZE_HINT};
#[cfg(stage1)]
#[cfg(stage2)]
use self::inst::{IMPL_T, EACH, SIZE_HINT};

impl<A> IMPL_T<A>: iter::BaseIter<A> {
    pure fn each(blk: fn(v: &A) -> bool) { EACH(&self, blk) }
    pure fn size_hint() -> Option<uint> { SIZE_HINT(&self) }
}

impl<A> IMPL_T<A>: iter::ExtendedIter<A> {
    pure fn eachi(blk: fn(uint, v: &A) -> bool) { iter::eachi(&self, blk) }
    pure fn all(blk: fn(&A) -> bool) -> bool { iter::all(&self, blk) }
    pure fn any(blk: fn(&A) -> bool) -> bool { iter::any(&self, blk) }
    pure fn foldl<B>(b0: B, blk: fn(&B, &A) -> B) -> B {
        iter::foldl(&self, move b0, blk)
    }
    pure fn position(f: fn(&A) -> bool) -> Option<uint> {
        iter::position(&self, f)
    }
}

impl<A: Eq> IMPL_T<A>: iter::EqIter<A> {
    pure fn contains(x: &A) -> bool { iter::contains(&self, x) }
    pure fn count(x: &A) -> uint { iter::count(&self, x) }
}

impl<A: Copy> IMPL_T<A>: iter::CopyableIter<A> {
    pure fn filter_to_vec(pred: fn(a: A) -> bool) -> ~[A] {
        iter::filter_to_vec(&self, pred)
    }
    pure fn map_to_vec<B>(op: fn(v: A) -> B) -> ~[B] {
        iter::map_to_vec(&self, op)
    }
    pure fn to_vec() -> ~[A] { iter::to_vec(&self) }

    pure fn flat_map_to_vec<B:Copy,IB:BaseIter<B>>(op: fn(a: A) -> IB)
        -> ~[B] {
        iter::flat_map_to_vec(&self, op)
    }

    pure fn find(p: fn(a: A) -> bool) -> Option<A> { iter::find(&self, p) }
}

impl<A: Copy Ord> IMPL_T<A>: iter::CopyableOrderedIter<A> {
    pure fn min() -> A { iter::min(&self) }
    pure fn max() -> A { iter::max(&self) }
}

