// edition:2018

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

struct T;

struct Tuple(i32);

struct Struct {
    a: i32
}

impl Struct {
    fn method(&self) {}
}

impl Future for Struct {
    type Output = Struct;
    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> { Poll::Pending }
}

impl Future for Tuple {
    type Output = Tuple;
    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> { Poll::Pending }
}

impl Future for T {
    type Output = Result<(), ()>;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

async fn foo() -> Result<(), ()> {
    Ok(())
}

async fn bar() -> Result<(), ()> {
    foo()?; //~ ERROR the `?` operator can only be applied to values that implement `Try`
    //~^ NOTE the `?` operator cannot be applied to type `impl Future<Output = Result<(), ()>>`
    //~| HELP the trait `Try` is not implemented for `impl Future<Output = Result<(), ()>>`
    //~| HELP consider `await`ing on the `Future`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    Ok(())
}

async fn struct_() -> Struct {
    Struct { a: 1 }
}

async fn tuple() -> Tuple {
    Tuple(1i32)
}

async fn baz() -> Result<(), ()> {
    let t = T;
    t?; //~ ERROR the `?` operator can only be applied to values that implement `Try`
    //~^ NOTE the `?` operator cannot be applied to type `T`
    //~| HELP the trait `Try` is not implemented for `T`
    //~| HELP consider `await`ing on the `Future`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`


    let _: i32 = tuple().0; //~ ERROR no field `0`
    //~^ HELP consider `await`ing on the `Future`
    //~| NOTE field not available in `impl Future`

    let _: i32 = struct_().a; //~ ERROR no field `a`
    //~^ HELP consider `await`ing on the `Future`
    //~| NOTE field not available in `impl Future`

    struct_().method(); //~ ERROR no method named
    //~^ NOTE method not found in `impl Future<Output = Struct>`
    //~| HELP consider `await`ing on the `Future`
    Ok(())
}

async fn match_() {
    match tuple() { //~ HELP consider `await`ing on the `Future`
        //~^ NOTE this expression has type `impl Future<Output = Tuple>`
        Tuple(_) => {} //~ ERROR mismatched types
        //~^ NOTE expected opaque type, found `Tuple`
        //~| NOTE expected opaque type `impl Future<Output = Tuple>`
    }
}

fn main() {}
