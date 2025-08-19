// Repro for <https://github.com/rust-lang/rust/issues/100013#issue-1323807923>.
//@ edition: 2021
//@ revisions: assumptions no_assumptions
//@[assumptions] compile-flags: -Zhigher-ranked-assumptions
//@[assumptions] check-pass
//@[no_assumptions] known-bug: #110338

#![feature(impl_trait_in_assoc_type)]

use std::future::Future;

pub trait FutureIterator: 'static {
    type Iterator;

    type Future<'s, 'cx>: Future<Output = Self::Iterator> + Send + 'cx
    where
        's: 'cx;

    fn get_iter<'s, 'cx>(&'s self, info: &'cx ()) -> Self::Future<'s, 'cx>;
}

trait IterCaller: 'static {
    type Future1<'cx>: Future<Output = ()> + Send + 'cx;
    type Future2<'cx>: Future<Output = ()> + Send + 'cx;

    fn call_1<'s, 'cx>(&'s self, cx: &'cx ()) -> Self::Future1<'cx>
    where
        's: 'cx;
    fn call_2<'s, 'cx>(&'s self, cx: &'cx ()) -> Self::Future2<'cx>
    where
        's: 'cx;
}

struct UseIter<FI1, FI2> {
    fi_1: FI1,
    fi_2: FI2,
}

impl<FI1, FI2> IterCaller for UseIter<FI1, FI2>
where
    FI1: FutureIterator + 'static + Send + Sync,
    for<'s, 'cx> FI1::Future<'s, 'cx>: Send,
    FI2: FutureIterator + 'static + Send + Sync,
{
    type Future1<'cx> = impl Future<Output = ()> + Send + 'cx
    where
        Self: 'cx;

    type Future2<'cx> = impl Future<Output = ()> + Send + 'cx
    where
        Self: 'cx;

    fn call_1<'s, 'cx>(&'s self, cx: &'cx ()) -> Self::Future1<'cx>
    where
        's: 'cx,
    {
        async {
            self.fi_1.get_iter(cx).await;
        }
    }

    fn call_2<'s, 'cx>(&'s self, cx: &'cx ()) -> Self::Future2<'cx>
    where
        's: 'cx,
    {
        async {
            self.fi_2.get_iter(cx).await;
        }
    }
}

fn main() {}
