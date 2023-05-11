// edition:2018
// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir
// #70935: Check if we do not emit snippet
// with newlines which lead complex diagnostics.

use std::future::Future;

async fn baz<T>(_c: impl FnMut() -> T) where T: Future<Output=()> {
}

fn foo(tx: std::sync::mpsc::Sender<i32>) -> impl Future + Send {
    //[no_drop_tracking]~^ ERROR future cannot be sent between threads safely
    //[drop_tracking,drop_tracking_mir]~^^ ERROR `Sender<i32>` cannot be shared between threads
    async move {
        baz(|| async{
            foo(tx.clone());
        }).await;
    }
}

fn bar(_s: impl Future + Send) {
}

fn main() {
    let (tx, _rx) = std::sync::mpsc::channel();
    bar(foo(tx));
}
