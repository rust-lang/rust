//@ known-bug: #131190
//@ compile-flags: -Cinstrument-coverage --edition=2018

use std::future::Future;

pub fn block_on<T>(fut: impl Future<Output = T>) -> T {}

async fn call_once(f: impl async FnOnce(DropMe)) {
    f(DropMe("world")).await;
}

struct DropMe(&'static str);

pub fn main() {
    block_on(async {
        let async_closure = async move |a: DropMe| {};
        call_once(async_closure).await;
    });
}
