// edition:2018

async fn hello() { //~ HELP try adding a return type
    0
    //~^ ERROR [E0308]
}

async fn world() -> () {
    0
    //~^ ERROR [E0308]
}

async fn suggest_await_in_async_fn_return() {
    hello()
    //~^ ERROR mismatched types [E0308]
    //~| HELP consider `await`ing on the `Future`
    //~| HELP consider using a semicolon here
    //~| SUGGESTION .await
}

fn main() {}
