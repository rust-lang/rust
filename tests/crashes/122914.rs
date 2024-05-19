//@ known-bug: #122914
use std::future::Future;
use std::pin::Pin;

impl<'a, F> Poll {
    fn project<'_>(self: Pin<&'pin mut Future>) -> Projection<'pin, 'a, F> {
        me.local_set.with(|| {
            let _ = self.poll(cx);
        })
    }
}
