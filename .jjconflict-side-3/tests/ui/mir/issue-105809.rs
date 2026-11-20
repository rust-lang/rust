// Non-regression test ICE from issue #105809 and duplicates.

//@ build-pass: the ICE is during codegen
//@ compile-flags: -Zmir-opt-level=1
//@ edition: 2018

use std::{future::Future, pin::Pin};

// Create a `T` without affecting analysis like `loop {}`.
fn create<T>() -> T {
    loop {}
}

async fn trivial_future() {}

struct Connection<H> {
    _h: H,
}

async fn complex_future<H>(conn: &Connection<H>) {
    let small_fut = async move {
        let _ = conn;
        trivial_future().await;
    };

    let mut tuple = (small_fut,);
    let (small_fut_again,) = &mut tuple;
    let _ = small_fut_again;
}

fn main() {
    let mut fut = complex_future(&Connection { _h: () });

    let mut cx = create();
    let future = unsafe { Pin::new_unchecked(&mut fut) };
    let _ = future.poll(&mut cx);
}
