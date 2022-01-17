// run-pass
// edition:2018

#![allow(unused)]

use std::future::Future;
use std::marker::PhantomData;

trait Stream {
    type Item;
}

struct Filter<St: Stream> {
    pending_item: St::Item,
}

fn filter<St: Stream>(_: St) -> Filter<St> {
    unimplemented!();
}

struct FilterMap<Fut, F> {
    f: F,
    pending: PhantomData<Fut>,
}

impl<Fut, F> Stream for FilterMap<Fut, F>
where
    F: FnMut() -> Fut,
    Fut: Future,
{
    type Item = ();
}

pub fn get_foo() -> impl Future + Send {
    async {
        let _y = &();
        let _x = filter(FilterMap {
            f: || async move { *_y },
            pending: PhantomData,
        });
        async {}.await;
    }
}

fn main() {}
