//@ edition:2018

async fn fun() {
    [1; ().await];
    //~^ error: `await` is only allowed inside `async` functions and blocks
}

fn main() {}
