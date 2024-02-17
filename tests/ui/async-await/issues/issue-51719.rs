//@ edition:2018
//
// Tests that the .await syntax can't be used to make a coroutine

async fn foo() {}

fn make_coroutine() {
    let _gen = || foo().await;
    //~^ ERROR `await` is only allowed inside `async` functions and blocks
}

fn main() {}
