//@ edition:2018
// #70935: Check if we do not emit snippet
// with newlines which lead complex diagnostics.

use std::future::Future;
use std::marker::PhantomData;

#[derive(Clone)]
struct NotSync(PhantomData<*mut ()>);
unsafe impl Send for NotSync {}

async fn baz<T>(_c: impl FnMut() -> T) where T: Future<Output=()> {
}

fn foo(x: NotSync) -> impl Future + Send {
    //~^ ERROR `*mut ()` cannot be shared between threads safely
    async move {
    //~^ ERROR `*mut ()` cannot be shared between threads safely
        baz(|| async {
            foo(x.clone());
        }).await;
    }
}

fn bar(_s: impl Future + Send) {
}

fn main() {
    let x = NotSync(PhantomData);
    bar(foo(x));
}
