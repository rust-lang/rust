//@ edition:2021

async fn foo() {
    inner::<false>().await
    //~^ ERROR: function takes 2 generic arguments but 1 generic argument was supplied
}

async fn inner<T, const PING: bool>() {}

fn main() {}
