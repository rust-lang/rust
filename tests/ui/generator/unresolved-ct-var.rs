// incremental
// edition:2021

fn main() {
    let _ = async {
        let s = std::array::from_fn(|_| ()).await;
        //~^ ERROR `[(); _]` is not a future
        //~| ERROR type inside `async` block must be known in this context
        //~| ERROR type inside `async` block must be known in this context
        //~| ERROR type inside `async` block must be known in this context
        //~| ERROR type inside `async` block must be known in this context
        //~| ERROR type inside `async` block must be known in this context
    };
}
