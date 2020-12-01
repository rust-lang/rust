// edition:2018
// #70935: Check if we do not emit snippet
// with newlines which lead complex diagnostics.

use std::future::Future;

async fn baz<T>(_c: impl FnMut() -> T) where T: Future<Output=()> {
}

fn foo(tx: std::sync::mpsc::Sender<i32>) -> impl Future + Send {
    //~^ ERROR: future cannot be sent between threads safely
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
