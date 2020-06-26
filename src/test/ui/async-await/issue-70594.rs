// edition:2018

async fn fun() {
    [1; ().await];
    //~^ error: `await` is only allowed inside `async` functions and blocks
    //~| error: `.await` is not allowed in a `const`
    //~| error: `loop` is not allowed in a `const`
    //~| error: `.await` is not allowed in a `const`
    //~| error: the trait bound `(): std::future::Future` is not satisfied
}

fn main() {}
