use std::future::Future;

async fn some_async_fn() {}

fn sync_side_effects() {}
fn custom() -> impl Future<Output = ()> {
    sync_side_effects();
    async {}
}

fn do_something_to_future(future: &mut impl Future<Output = ()>) {}

fn main() {
    let _ = some_async_fn();
    let _ = custom();

    let mut future = some_async_fn();
    do_something_to_future(&mut future);
    let _ = future;
}
