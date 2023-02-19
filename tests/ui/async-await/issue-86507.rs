// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir
// edition:2018

use ::core::pin::Pin;
use ::core::future::Future;
use ::core::marker::Send;

trait Foo {
    fn bar<'me, 'async_trait, T: Send>(x: &'me T)
        -> Pin<Box<dyn Future<Output = ()> + Send + 'async_trait>>
        where 'me: 'async_trait;
}

impl Foo for () {
    fn bar<'me, 'async_trait, T: Send>(x: &'me T)
        -> Pin<Box<dyn Future<Output = ()> + Send + 'async_trait>>
        where 'me:'async_trait {
            Box::pin( //~ ERROR future cannot be sent between threads safely
                async move {
                    let x = x;
                }
            )
         }
}

fn main() { }
