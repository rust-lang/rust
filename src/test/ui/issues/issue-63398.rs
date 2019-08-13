// edition:2018
#![feature(async_await)]

async fn do_the_thing() -> u8 {
    8
}

fn main() {
    let x = move || {};
    let y = do_the_thing().await; //~ ERROR `await` is only allowed inside `async` functions
}
