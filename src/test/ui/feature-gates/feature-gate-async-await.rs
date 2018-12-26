// edition:2018

#![feature(futures_api)]

async fn foo() {} //~ ERROR async fn is unstable

fn main() {
    let _ = async {}; //~ ERROR async blocks are unstable
    let _ = async || {}; //~ ERROR async closures are unstable
}
