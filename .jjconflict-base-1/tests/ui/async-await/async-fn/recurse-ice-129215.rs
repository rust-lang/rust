//@ edition: 2021

async fn a() {
    //~^ ERROR `()` is not a future
    //~| ERROR mismatched types
    a() //~ ERROR `()` is not a future
}

fn main() {}
