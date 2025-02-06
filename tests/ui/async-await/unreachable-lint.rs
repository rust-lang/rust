//@ edition:2018
#![deny(unreachable_code)]

async fn foo() {
    endless().await;
    //~^ ERROR unreachable expression
}

async fn endless() -> ! {
    loop {}
}

fn main() {}
