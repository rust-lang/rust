//@ check-pass
//@ edition:2021

use core::future::Future;

fn main() {
    do_async(async { (0,) }, {
        // closure must be inside block
        |info| println!("{:?}", info.0)
    });
}

fn do_async<R, Fut, F>(_tokio_fut: Fut, _glib_closure: F)
where
    Fut: Future<Output = R>,
    F: FnOnce(R),
{
}
