//@ check-pass
//@ edition:2018
#![deny(unreachable_code)]

async fn foo() {
    endless().await;
}

async fn endless() -> ! {
    loop {}
}

fn main() { }
