// edition:2018
// revisions: normal drop_tracking
// [drop_tracking]compile-flags:-Zdrop-tracking
// #70935: Check if we do not emit snippet
// with newlines which lead complex diagnostics.

use std::future::Future;

async fn baz<T>(_c: impl FnMut() -> T) where T: Future<Output=()> {
//[drop_tracking]~^ within this async block
}

fn foo(tx: std::sync::mpsc::Sender<i32>) -> impl Future + Send {
    //[normal]~^ ERROR: future cannot be sent between threads safely
    //[drop_tracking]~^^ ERROR: `Sender<i32>` cannot be shared
    //[drop_tracking]~|  NOTE: cannot be shared
    //[drop_tracking]~|  NOTE: requirements on the impl of `Send`
    //[drop_tracking]~|  NOTE: captures the following types
    //[drop_tracking]~|  NOTE: in this expansion
    //[drop_tracking]~|  NOTE: in this expansion
    //[drop_tracking]~|  NOTE: in this expansion
    //[drop_tracking]~|  NOTE: in this expansion
    async move {
    //[drop_tracking]~^ within this async block
        baz(|| async{ //[drop_tracking]~ NOTE: used within this closure
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
