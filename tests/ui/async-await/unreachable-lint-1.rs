//@ edition:2018
#![deny(unreachable_code)]

async fn foo() {
    return; bar().await;
    //~^ ERROR unreachable statement
}

async fn bar() {
}

fn main() { }
