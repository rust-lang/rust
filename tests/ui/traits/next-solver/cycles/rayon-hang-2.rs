//@ compile-flags: -Znext-solver
//@ check-pass

// A regression test for trait-system-refactor-initiative#109.
// Unlike `rayon-hang-1.rs` the cycles in this test are not
// unproductive, which causes the `AliasRelate` goal when trying
// to apply where-clauses to only error in the second iteration.
//
// This makes the exponential blowup to be significantly harder
// to avoid.

trait ParallelIterator: Sized {
    type Item;
}

trait IntoParallelIteratorIndir {
    type Iter: ParallelIterator<Item = Self::Item>;
    type Item;
}
impl<I> IntoParallelIteratorIndir for I
where
    Box<I>: IntoParallelIterator,
{
    type Iter = <Box<I> as IntoParallelIterator>::Iter;
    type Item = <Box<I> as IntoParallelIterator>::Item;
}
trait IntoParallelIterator {
    type Iter: ParallelIterator<Item = Self::Item>;
    type Item;
}
impl<T: ParallelIterator> IntoParallelIterator for T {
    type Iter = T;
    type Item = T::Item;
}

macro_rules! multizip_impls {
    ($($T:ident),+) => {
       fn foo<'a, $( $T, )+>() where
        $(
            $T: IntoParallelIteratorIndir,
            $T::Iter: ParallelIterator,
        )+
        {}
    }
}

multizip_impls! { A, B, C, D, E, F, G, H, I, J, K, L }

fn main() {}
