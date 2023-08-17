// compile-flags: -Ztrait-solver=next -Zdrop-tracking-mir
// edition: 2021
// revisions: pass fail
//[pass] check-fail
// WARN new-solver BUG.

#![feature(negative_impls)]

struct NotSync;
impl !Sync for NotSync {}

async fn foo() {
//[pass]~^ ERROR type mismatch
    #[cfg(pass)]
    let x = &();
    #[cfg(fail)]
    let x = &NotSync;
    bar().await;
    #[allow(dropping_references)]
    drop(x);
}

async fn bar() {}
//[pass]~^ ERROR type mismatch

fn main() {
    fn is_send(_: impl Send) {}
    is_send(foo());
    //[fail]~^ ERROR `impl Future<Output = ()>` cannot be sent between threads safely
}
