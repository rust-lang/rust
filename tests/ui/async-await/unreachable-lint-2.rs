//@ edition:2018

#![deny(unreachable_code)]

async fn foo() {
    endless().await;
    println!("this is unreachable!");
    //~^ ERROR unreachable statement
}

async fn endless() -> ! {
    loop {}
}

fn main() { }
