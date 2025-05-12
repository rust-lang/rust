//@ edition:2018
//@ run-pass

use std::future::Future;

trait AsyncCallback<'a> {
    type Out;
}

impl<'a, Fut, T, F> AsyncCallback<'a> for F
where
    F: FnOnce(&'a mut ()) -> Fut,
    Fut: Future<Output = T> + Send + 'a,
{
    type Out = T;
}

trait CallbackMarker {}

impl<F, T> CallbackMarker for F
where
    T: 'static,
    for<'a> F: AsyncCallback<'a, Out = T> + Send,
{
}

fn do_sth<F: CallbackMarker>(_: F) {}

async fn callback(_: &mut ()) -> impl Send {}

fn main() {
    do_sth(callback);
}
