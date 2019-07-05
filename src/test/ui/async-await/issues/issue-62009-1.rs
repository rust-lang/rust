// edition:2018

#![feature(async_await)]

async fn print_dur() {}

fn main() {
    async { let (); }.await;
    //~^ ERROR `await` is only allowed inside `async` functions and blocks
    async {
    //~^ ERROR `await` is only allowed inside `async` functions and blocks
        let task1 = print_dur().await;
    }.await;
    (|_| 2333).await;
    //~^ ERROR `await` is only allowed inside `async` functions and blocks
    //~^^ ERROR
}
