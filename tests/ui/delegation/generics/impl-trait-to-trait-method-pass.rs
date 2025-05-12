//@ run-pass

#![feature(fn_delegation)]
#![allow(incomplete_features)]

use std::iter::{Iterator, Map};

pub mod same_trait {
    use super::*;

    pub struct MapOuter<I, F> {
        pub inner: Map<I, F>
    }

    impl<B, I: Iterator, F> Iterator for MapOuter<I, F>
    where
        F: FnMut(I::Item) -> B,
    {
        type Item = <Map<I, F> as Iterator>::Item;

        reuse Iterator::{next, fold} { self.inner }
    }
}
use same_trait::MapOuter;

mod another_trait {
    use super::*;

    trait ZipImpl<A, B> {
        type Item;

        fn next(&mut self) -> Option<Self::Item>;
    }

    pub struct Zip<A, B> {
        pub a: A,
        pub b: B,
    }

    impl<A: Iterator, B: Iterator> ZipImpl<A, B> for Zip<A, B> {
        type Item = (A::Item, B::Item);

        fn next(&mut self) -> Option<(A::Item, B::Item)> {
            let x = self.a.next()?;
            let y = self.b.next()?;
            Some((x, y))
        }
    }

    impl<A: Iterator, B: Iterator> Iterator for Zip<A, B> {
        type Item = (A::Item, B::Item);

        // Parameters are inherited from `Iterator::next`, not from `ZipImpl::next`.
        // Otherwise, there would be a compilation error due to an unconstrained parameter.
        reuse ZipImpl::next;
    }
}
use another_trait::Zip;

fn main() {
    {
        let x = vec![1, 2, 3];
        let iter = x.iter().map(|val| val * 2);
        let outer_iter = MapOuter { inner: iter };
        let val = outer_iter.fold(0, |acc, x| acc + x);
        assert_eq!(val, 12);
    }

    {
        let x = vec![1, 2];
        let y = vec![4, 5];

        let mut zip = Zip { a: x.iter(), b: y.iter() };
        assert_eq!(zip.next(), Some((&1, &4)));
        assert_eq!(zip.next(), Some((&2, &5)));
    }
}
