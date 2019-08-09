// edition:2018

#![feature(async_await)]

async fn inc(limit: i64) -> i64 {
    limit + 1
}

fn main() {
    let result = inc(10000);
    let finished = result.await;
    //~^ ERROR `await` is only allowed inside `async` functions and blocks
}
