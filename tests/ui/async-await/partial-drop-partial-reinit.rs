//@ edition:2021
#![feature(negative_impls)]
#![allow(unused)]

fn main() {
    gimme_send(foo());
    //~^ ERROR future cannot be sent between threads safely
    //~| NOTE future returned by `foo` is not `Send`
}

fn gimme_send<T: Send>(t: T) {
    //~^ NOTE required by this bound in `gimme_send`
    //~| NOTE required by a bound in `gimme_send`
    drop(t);
}

struct NotSend {}
//~^ HELP within `impl Future<Output = ()>`, the trait `Send` is not implemented for `NotSend`

impl Drop for NotSend {
    fn drop(&mut self) {}
}

impl !Send for NotSend {}

async fn foo() {
    let mut x = (NotSend {},);
    //~^ NOTE has type `UnsafePinned<(NotSend,)>` which is not `Send`
    drop(x.0);
    x.0 = NotSend {};
    bar().await;
    //~^ NOTE future is not `Send` as this value is used across an await
    //~| NOTE await occurs here, with `mut x` maybe used later
    //~| NOTE in this expansion of desugaring of `await` expression
}

async fn bar() {}
