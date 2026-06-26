//@ check-fail
//@ edition: 2021

use std::future::Future;
use std::pin::pin;
use std::task::*;

pub fn block_on<T>(fut: impl Future<Output = T>) -> T {
    let mut fut = pin!(fut);
    // Poll loop, just to test the future...
    let ctx = &mut Context::from_waker(Waker::noop());

    loop {
        match fut.as_mut().poll(ctx) {
            Poll::Pending => {}
            Poll::Ready(t) => break t,
        }
    }
}

trait Blah {
    async fn iter<T>(&mut self, iterator: T)
    where
        T: IntoIterator<Item = ()>;
}

impl Blah for () {
    async fn iter<T>(&mut self, iterator: T)
    //~^ ERROR recursion in an async fn requires boxing
    where
        T: IntoIterator<Item = ()>,
    {
        Blah::iter(self, iterator).await
    }
}

struct Wrap<T: Blah> {
    t: T,
}

impl<T: Blah> Wrap<T>
where
    T: Blah,
{
    async fn ice(&mut self) {
        let arr: [(); 0] = [];
        self.t.iter(arr.into_iter()).await;
    }
}

fn main() {
    block_on(async {
        let mut t = Wrap { t: () };
        t.ice();
    })
}
