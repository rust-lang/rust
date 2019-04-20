// edition:2018

#![feature(futures_api)]

struct S;

impl S {
    async fn foo() {} //~ ERROR async fn is unstable
}

trait T {
    async fn foo(); //~ ERROR trait fns cannot be declared `async`
    //~^ ERROR async fn is unstable
}

async fn foo() {} //~ ERROR async fn is unstable

fn main() {
    let _ = async {}; //~ ERROR async blocks are unstable
    let _ = async || {}; //~ ERROR async closures are unstable
}
