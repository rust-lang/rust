// compile-flags: -Ztrait-solver=next
// edition: 2021
// revisions: pass fail
//[pass] check-pass

#![feature(negative_impls)]

struct NotSync;
impl !Sync for NotSync {}

async fn foo() {
    #[cfg(pass)]
    let x = &();
    #[cfg(fail)]
    let x = &NotSync;
    bar().await;
    #[allow(dropping_references)]
    drop(x);
}

async fn bar() {}

fn main() {
    fn is_send(_: impl Send) {}
    is_send(foo());
    //[fail]~^ ERROR `impl Future<Output = ()>` cannot be sent between threads safely
}
