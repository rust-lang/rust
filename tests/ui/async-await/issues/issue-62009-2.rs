//@ edition:2018

async fn print_dur() {}

fn main() {
    (async || 2333)().await;
    //~^ ERROR `await` is only allowed inside `async` functions and blocks
}
