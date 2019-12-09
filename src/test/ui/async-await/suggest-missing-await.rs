// edition:2018
// run-rustfix

fn take_u32(_x: u32) {}

async fn make_u32() -> u32 {
    22
}

#[allow(unused)]
async fn suggest_await_in_async_fn() {
    let x = make_u32();
    take_u32(x)
    //~^ ERROR mismatched types [E0308]
    //~| HELP consider using `.await` here
    //~| SUGGESTION x.await
}

async fn dummy() {}

#[allow(unused)]
async fn suggest_await_in_async_fn_return() {
    dummy()
    //~^ ERROR mismatched types [E0308]
    //~| HELP try adding a semicolon
    //~| HELP consider using `.await` here
    //~| SUGGESTION dummy().await
}

fn main() {}
