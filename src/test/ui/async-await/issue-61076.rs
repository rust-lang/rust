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

    let _: i32 = tuple().0; //~ ERROR no field `0`

    let _: i32 = struct_().a; //~ ERROR no field `a`

    struct_().method(); //~ ERROR no method named

    Ok(())
}

async fn match_() {
    match tuple() {
        Tuple(_) => {} //~ ERROR mismatched types
    }
}

fn main() {}
