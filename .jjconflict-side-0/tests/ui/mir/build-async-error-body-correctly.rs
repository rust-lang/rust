//@ edition: 2021

async fn asyncfn() {
    let binding = match true {};
    //~^ ERROR non-exhaustive patterns: type `bool` is non-empty
}

fn main() {}
