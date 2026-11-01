//@ edition:2018

async fn print_dur() {}

fn main() {
    async { let (); }.await;
    //~^ ERROR `await` is only allowed inside `async` functions and blocks
    async {
        let task1 = print_dur().await;
    }.await;
    //~^ ERROR `await` is only allowed inside `async` functions and blocks
    (|_| 2333).await;
    //~^ ERROR `await` is only allowed inside `async` functions and blocks
}
