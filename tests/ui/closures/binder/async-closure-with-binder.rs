// edition:2021
#![feature(closure_lifetime_binder)]
#![feature(async_closure)]
fn main() {
    for<'a> async || ();
    //~^ ERROR `for<...>` binders on `async` closures are not currently supported
    //~^^ ERROR implicit types in closure signatures are forbidden when `for<...>` is present
}
