//@ edition: 2021

macro_rules! x {
    ($x:item) => {}
}

x! {
    async fn foo() -> impl async Fn() { }
    //~^ ERROR `async` trait bounds are unstable
}

fn main() {}
