//@ check-pass
//@ compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

// Minimization of a hang in rayon, cc trait-solver-refactor-initiative#109

pub trait ParallelIterator {
    type Item;
}

macro_rules! multizip_impl {
    ($($T:ident),+) => {
        impl<$( $T, )+> ParallelIterator for ($( $T, )+)
        where
            $(
                $T: ParallelIterator,
                $T::Item: ParallelIterator,
            )+
        {
            type Item = ();
        }
    }
}

multizip_impl! { A, B, C, D, E, F, G, H, I, J, K, L }

fn main() {}
