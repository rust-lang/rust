// Regression test for borrowck ICE #133252
//@ edition:2021
use std::future::Future;

trait Owned: 'static {}
fn ice() -> impl Future<Output = &'static dyn Owned> {
    async {
        let not_static = 0;
        force_send(async_load(&not_static));
        //~^ ERROR implementation of `LoadQuery` is not general enough
        //~| ERROR `not_static` does not live long enough
        loop {}
    }
}

fn force_send<T: Send>(_: T) {}

fn async_load<'a, T: LoadQuery<'a>>(this: T) -> impl Future {
    async {
        this.get_future().await;
    }
}

trait LoadQuery<'a>: Sized {
    type LoadFuture: Future;

    fn get_future(self) -> Self::LoadFuture {
        loop {}
    }
}

impl<'a> LoadQuery<'a> for &'a u8 {
    type LoadFuture = SimpleFuture;
}

struct SimpleFuture;
impl Future for SimpleFuture {
    type Output = ();
    fn poll(
        self: std::pin::Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        loop {}
    }
}

fn main() {}
