// This makes use of a clever hack that brson came up with to
// workaround our lack of traits and lack of macros.  See core.{rc,rs} for
// how this file is used.

import inst::{IMPL_T, EACH, SIZE_HINT};
export extensions;

impl extensions<A> of iter::base_iter<A> for IMPL_T<A> {
    fn each(blk: fn(A) -> bool) { EACH(self, blk) }
    fn size_hint() -> option<uint> { SIZE_HINT(self) }
    fn eachi(blk: fn(uint, A) -> bool) { iter::eachi(self, blk) }
    fn all(blk: fn(A) -> bool) -> bool { iter::all(self, blk) }
    fn any(blk: fn(A) -> bool) -> bool { iter::any(self, blk) }
    fn foldl<B>(+b0: B, blk: fn(B, A) -> B) -> B {
        iter::foldl(self, b0, blk)
    }
    fn contains(x: A) -> bool { iter::contains(self, x) }
    fn count(x: A) -> uint { iter::count(self, x) }
}

impl extensions<A:copy> for IMPL_T<A> {
    fn filter_to_vec(pred: fn(A) -> bool) -> [A] {
        iter::filter_to_vec(self, pred)
    }
    fn map_to_vec<B>(op: fn(A) -> B) -> [B] { iter::map_to_vec(self, op) }
    fn to_vec() -> [A] { iter::to_vec(self) }

    // FIXME--bug in resolve prevents this from working (#2611)
    // fn flat_map_to_vec<B:copy,IB:base_iter<B>>(op: fn(A) -> IB) -> [B] {
    //     iter::flat_map_to_vec(self, op)
    // }

    fn min() -> A { iter::min(self) }
    fn max() -> A { iter::max(self) }
}
