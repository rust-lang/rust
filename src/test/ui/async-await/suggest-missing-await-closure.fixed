// edition:2018
// run-rustfix

#![feature(async_closure)]

fn take_u32(_x: u32) {}

async fn make_u32() -> u32 {
    22
}

#[allow(unused)]
async fn suggest_await_in_async_closure() {
    async || {
        let x = make_u32();
        take_u32(x.await)
        //~^ ERROR mismatched types [E0308]
        //~| HELP consider `await`ing on the `Future`
        //~| SUGGESTION .await
    };
}

fn main() {}
