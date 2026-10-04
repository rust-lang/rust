//@ compile-flags: -Znext-solver
//@ check-pass

// A regression test for trait-system-refactor-initiative#109.

trait ParallelIterator: Sized {
    type Item;
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
       fn foo<$( $T, )+>() where
        $(
            $T: IntoParallelIterator,
            $T::Iter: ParallelIterator,
        )+
            ($( $T, )+): IntoParallelIterator<Item = ($( $T::Item, )+)>,
        {}
    }
}

multizip_impls! { A, B, C, D, E, F, G, H, I, J, K, L }

fn main() {}
