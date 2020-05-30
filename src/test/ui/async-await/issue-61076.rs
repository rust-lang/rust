// edition:2018

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

struct T;

struct UnionStruct(i32);

struct Struct {
    a: i32
}

enum Enum {
    A
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
    foo()?; //~ ERROR the `?` operator can only be applied to values that implement `std::ops::Try`
    Ok(())
}

async fn baz() -> Result<(), ()> {
    let t = T;
    t?; //~ ERROR the `?` operator can only be applied to values that implement `std::ops::Try`

    let _: i32 = async {
        UnionStruct(1i32)
    }.0; //~ ERROR no field `0`

    let _: i32 = async {
        Struct { a: 1i32 }
    }.a; //~ ERROR no field `a`

    if let Enum::A = async { Enum::A } {} //~ ERROR mismatched type

    Ok(())
}


fn main() {}
