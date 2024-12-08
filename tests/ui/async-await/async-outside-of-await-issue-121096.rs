//@ edition:2021

fn main() {
    async {
        use std::ops::Add;
        let _ = 1.add(3);
    }.await
    //~^ ERROR `await` is only allowed inside `async` functions and blocks
}
